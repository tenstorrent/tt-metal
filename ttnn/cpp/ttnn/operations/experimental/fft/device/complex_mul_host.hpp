// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// complex_mul_host.hpp — Host launcher for the elementwise complex-multiply
// kernel (`kernels/{compute,dataflow}/complex_mul_*`).
//
// Purpose
// -------
// The Bluestein backend in `universal_host.hpp` previously did the chirp
// pre- and post-multiplies as tight host C++ loops on `std::vector<Complex>`.
// With `TT_FFT_DEVICE_CHIRP_MUL=1`, those loops are replaced by a dispatch
// through this header so the per-element complex multiplies execute on
// Tensix cores (SFPU fp32, "precise" math) instead of the host CPU.
//
// Geometry
// --------
// Inputs are tile-major fp32, real and imag in separate buffers
// (`kTileElems = 1024` fp32 values per tile, 4 KiB).
//
//   * A is `num_a_tiles` long.
//   * B is `num_b_tiles` long and BROADCASTS:
//         b_tile_idx = a_tile_idx mod num_b_tiles
//     so the chirp (length M, => M/1024 tiles) can be reused across every
//     row of A without re-uploading.
//   * OUT shape matches A.
//
// Constraints (caller responsibility)
//   * `num_a_tiles >= 1`.
//   * `num_b_tiles >= 1` and `num_a_tiles % num_b_tiles == 0`.
//     (Caller should pick a tile partition where each input row is a
//     whole multiple of the chirp tile count.)
//
// Buffer ownership
//   * A and OUT device buffers are owned by the ComplexMulPlan.
//   * B device buffers are PROVIDED by the caller (typically lives in
//     `BluesteinPlan` so the chirp is uploaded once per N).
//
// Plan cache
//   * Keyed by `(MeshDevice*, num_a_tiles, num_b_tiles, b_r_buf*, b_i_buf*)`.
//   * Reusing the same (count, M, BluesteinPlan) tuple will hit the cache
//     and skip program build on every call after the first.

#pragma once

#include "fft_inner_host.hpp"   // make_mesh_buf, buf_addr
#include "stockham_host.hpp"    // pick_batch_grid, max_cores_for_grid, batch_logical_core,
                                // kTileElems, kTileSizeFp32

#include "tt-metalium/host_api.hpp"
#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/mesh_command_queue.hpp"
#include "tt-metalium/mesh_workload.hpp"
#include "tt-metalium/mesh_buffer.hpp"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fft_complex_mul {

using Complex = std::complex<float>;
using tt::tt_metal::distributed::MeshDevice;
using tt::tt_metal::distributed::MeshBuffer;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshCoordinateRange;
using tt::tt_metal::distributed::MeshCommandQueue;
using tt::tt_metal::distributed::MeshWorkload;
using tt::tt_metal::distributed::WriteShard;
using tt::tt_metal::distributed::ReadShard;
using tt::tt_metal::distributed::EnqueueMeshWorkload;

using fft_example::make_mesh_buf;
using fft_example::buf_addr;
using fft_example::kTileElems;
using fft_example::kTileSizeFp32;

// ───────────────────────── toggle ──────────────────────────────────────────

// Honour an environment variable so the new code path is opt-in. Checked
// once per process and cached.
inline bool device_chirp_mul_enabled() {
    static const bool v = [] {
        if (const char* s = std::getenv("TT_FFT_DEVICE_CHIRP_MUL")) {
            return std::string(s) != "0" && !std::string(s).empty();
        }
        return false;
    }();
    return v;
}

// ───────────────────────── plan struct ─────────────────────────────────────

struct ComplexMulPlan {
    uint32_t num_a_tiles  = 0;
    uint32_t num_b_tiles  = 0;
    uint32_t num_cores    = 0;
    uint32_t tiles_per_core = 0;
    uint32_t grid_cols    = 0;
    uint32_t grid_rows    = 0;

    std::shared_ptr<MeshDevice> md;
    std::shared_ptr<MeshBuffer> a_r_buf, a_i_buf;
    std::shared_ptr<MeshBuffer> out_r_buf, out_i_buf;
    std::shared_ptr<MeshBuffer> b_r_buf, b_i_buf;   // held to keep alive
    MeshWorkload                workload;

    // Reused host scratch — pre-zeroed at plan build time. Caller writes
    // into [0, count*M); padded region stays zero so the device gets
    // well-defined data.
    std::vector<float> a_r_host, a_i_host;
    std::vector<float> out_r_host, out_i_host;

    // Interleaved complex staging for Bluestein chirp pre/post mul (length
    // num_a_tiles * kTileElems). Allocated once per cached plan — not per call.
    std::vector<Complex> a_padded;
    std::vector<Complex> out_padded_M;

    bool initialized = false;
};

// ───────────────────────── plan cache ──────────────────────────────────────

namespace detail {

struct PlanKey {
    MeshDevice* md;
    uint32_t    num_a_tiles;
    uint32_t    num_b_tiles;
    MeshBuffer* b_r_buf;
    MeshBuffer* b_i_buf;
    bool operator==(const PlanKey& o) const noexcept {
        return md == o.md
            && num_a_tiles == o.num_a_tiles
            && num_b_tiles == o.num_b_tiles
            && b_r_buf == o.b_r_buf
            && b_i_buf == o.b_i_buf;
    }
};

struct PlanKeyHash {
    size_t operator()(const PlanKey& k) const noexcept {
        const uint64_t m = reinterpret_cast<uint64_t>(k.md);
        const uint64_t r = reinterpret_cast<uint64_t>(k.b_r_buf);
        const uint64_t i = reinterpret_cast<uint64_t>(k.b_i_buf);
        uint64_t h = m;
        h ^= 0x9E3779B97F4A7C15ULL * k.num_a_tiles;
        h ^= 0xC2B2AE3D27D4EB4FULL * k.num_b_tiles;
        h ^= 0x165667B19E3779F9ULL * r;
        h ^= 0x85EBCA77C2B2AE63ULL * i;
        return static_cast<size_t>(h);
    }
};

inline std::unordered_map<PlanKey, std::shared_ptr<ComplexMulPlan>, PlanKeyHash>&
plan_cache() {
    static std::unordered_map<PlanKey, std::shared_ptr<ComplexMulPlan>, PlanKeyHash> c;
    return c;
}

// Pick the largest divisor of num_a_tiles that fits the grid. Always ≥ 1.
inline uint32_t pick_num_cores(uint32_t num_a_tiles, uint32_t max_cores) {
    uint32_t nc = std::min(num_a_tiles, max_cores);
    while (nc > 1 && num_a_tiles % nc != 0u) --nc;
    return nc;
}

}  // namespace detail

// ───────────────────────── plan builder ────────────────────────────────────

inline std::shared_ptr<ComplexMulPlan> make_complex_mul_plan(
    std::shared_ptr<MeshDevice>      md,
    uint32_t                         num_a_tiles,
    uint32_t                         num_b_tiles,
    std::shared_ptr<MeshBuffer>      b_r_buf,
    std::shared_ptr<MeshBuffer>      b_i_buf)
{
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;

    auto pp = std::make_shared<ComplexMulPlan>();
    pp->md           = md;
    pp->num_a_tiles  = num_a_tiles;
    pp->num_b_tiles  = num_b_tiles;
    pp->b_r_buf      = b_r_buf;
    pp->b_i_buf      = b_i_buf;

    assert(num_a_tiles >= 1u);
    assert(num_b_tiles >= 1u);

    // Core split: pick max divisor of num_a_tiles that fits the grid, and
    // also a usable cols × rows layout for it.
    const auto     dev_grid  = md->compute_with_storage_grid_size();
    const uint32_t max_cores = fft_stockham::max_cores_for_grid(dev_grid.x, dev_grid.y);
    pp->num_cores      = detail::pick_num_cores(num_a_tiles, max_cores);
    pp->tiles_per_core = num_a_tiles / pp->num_cores;
    assert(pp->num_cores * pp->tiles_per_core == num_a_tiles);
    std::tie(pp->grid_cols, pp->grid_rows) =
        fft_stockham::pick_batch_grid(pp->num_cores, dev_grid.x);

    // ── A + OUT device buffers ────────────────────────────────────────────
    const uint32_t a_bytes = num_a_tiles * kTileSizeFp32;
    pp->a_r_buf   = make_mesh_buf(md, a_bytes, kTileSizeFp32);
    pp->a_i_buf   = make_mesh_buf(md, a_bytes, kTileSizeFp32);
    pp->out_r_buf = make_mesh_buf(md, a_bytes, kTileSizeFp32);
    pp->out_i_buf = make_mesh_buf(md, a_bytes, kTileSizeFp32);

    // ── Host scratch (re-used across calls) ───────────────────────────────
    const size_t scratch_floats = static_cast<size_t>(num_a_tiles) * kTileElems;
    pp->a_r_host.assign(scratch_floats, 0.0f);
    pp->a_i_host.assign(scratch_floats, 0.0f);
    pp->out_r_host.assign(scratch_floats, 0.0f);
    pp->out_i_host.assign(scratch_floats, 0.0f);

    const size_t scratch_complex = scratch_floats;
    pp->a_padded.assign(scratch_complex, Complex{0.0f, 0.0f});
    pp->out_padded_M.assign(scratch_complex, Complex{0.0f, 0.0f});

    // ── Build the program ─────────────────────────────────────────────────
    Program prog = CreateProgram();

    const CoreCoord first{0, 0};
    const CoreCoord last{pp->grid_cols - 1u, pp->grid_rows - 1u};
    const CoreRange cr(first, last);

    constexpr uint32_t kNumCbs       = 8u;
    constexpr uint32_t kCbTiles[8]   = {2, 2, 2, 2, 2, 2, 1, 1};  // A_R, A_I, B_R, B_I, OUT_R, OUT_I, TMP_R, TMP_I

    for (uint32_t id = 0; id < kNumCbs; ++id) {
        CircularBufferConfig c(
            kCbTiles[id] * kTileSizeFp32,
            {{id, tt::DataFormat::Float32}});
        c.set_page_size(id, kTileSizeFp32);
        CreateCircularBuffer(prog, cr, c);
    }

    auto rk = CreateKernel(
        prog,
        "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/complex_mul_reader.cpp",
        cr,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc       = NOC::RISCV_0_default});

    auto wk = CreateKernel(
        prog,
        "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/complex_mul_writer.cpp",
        cr,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc       = NOC::RISCV_1_default});

    std::vector<UnpackToDestMode> u2d(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (uint32_t id = 0; id < kNumCbs; ++id) {
        u2d[id] = UnpackToDestMode::UnpackToDestFp32;
    }

    auto ck = CreateKernel(
        prog,
        "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/compute/complex_mul_compute.cpp",
        cr,
        ComputeConfig{
            .math_fidelity       = MathFidelity::HiFi4,
            .fp32_dest_acc_en    = true,
            .unpack_to_dest_mode = u2d});

    for (uint32_t c = 0; c < pp->num_cores; ++c) {
        const CoreCoord logical = fft_stockham::batch_logical_core(c, pp->grid_cols);
        const uint32_t  base    = c * pp->tiles_per_core;

        SetRuntimeArgs(prog, rk, logical, {
            buf_addr(pp->a_r_buf), buf_addr(pp->a_i_buf),
            buf_addr(pp->b_r_buf), buf_addr(pp->b_i_buf),
            base, pp->tiles_per_core, pp->num_b_tiles,
        });
        SetRuntimeArgs(prog, wk, logical, {
            buf_addr(pp->out_r_buf), buf_addr(pp->out_i_buf),
            base, pp->tiles_per_core,
        });
        SetRuntimeArgs(prog, ck, logical, {pp->tiles_per_core});
    }

    pp->workload.add_program(
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 0)),
        std::move(prog));
    pp->initialized = true;
    return pp;
}

inline std::shared_ptr<ComplexMulPlan> get_cached_complex_mul_plan(
    std::shared_ptr<MeshDevice> md,
    uint32_t                    num_a_tiles,
    uint32_t                    num_b_tiles,
    std::shared_ptr<MeshBuffer> b_r_buf,
    std::shared_ptr<MeshBuffer> b_i_buf)
{
    detail::PlanKey key{
        md.get(), num_a_tiles, num_b_tiles, b_r_buf.get(), b_i_buf.get()};
    auto& cache = detail::plan_cache();
    if (auto it = cache.find(key); it != cache.end()) return it->second;
    auto pp = make_complex_mul_plan(md, num_a_tiles, num_b_tiles, b_r_buf, b_i_buf);
    cache.emplace(key, pp);
    return pp;
}

// ───────────────────────── helpers ─────────────────────────────────────────

// Upload a complex vector of length `count * complex_len_per_row` (interleaved
// `Complex`), packed into the device's tile-major layout. Caller has already
// zero-padded `src` to a multiple of `kTileElems` per row.
inline void deinterleave_to_planes(
    const Complex* src,
    size_t          total_complex,
    std::vector<float>& real,
    std::vector<float>& imag)
{
    assert(real.size() >= total_complex);
    assert(imag.size() >= total_complex);
    for (size_t i = 0; i < total_complex; ++i) {
        real[i] = src[i].real();
        imag[i] = src[i].imag();
    }
    // Anything beyond `total_complex` stays at whatever the host scratch was
    // last written to; for the chirp-mul use case the caller pre-zeroes the
    // scratch in `assign(_, 0)` at plan build, then only ever writes the
    // active prefix back, so any tail is still zero across calls.
}

inline void interleave_from_planes(
    Complex* dst,
    size_t   total_complex,
    const std::vector<float>& real,
    const std::vector<float>& imag)
{
    assert(real.size() >= total_complex);
    assert(imag.size() >= total_complex);
    for (size_t i = 0; i < total_complex; ++i) {
        dst[i] = Complex{real[i], imag[i]};
    }
}

// ───────────────────────── run helper ──────────────────────────────────────
//
// Runs the cached complex-mul plan once.
//
//   out[i] = a_padded[i] * b[i mod (num_b_tiles*kTileElems)]   for i in
//   [0, count*M).
//
// `a_padded` length = num_a_tiles * kTileElems   (caller zero-pads to tile
//                                                 boundary)
// `out_padded` length = num_a_tiles * kTileElems (caller-provided, will be
//                                                 overwritten)
//
// Returns immediately (sync inside ReadShard).
inline void run_complex_mul(
    std::shared_ptr<ComplexMulPlan>& plan,
    const Complex* a_padded,
    Complex*       out_padded)
{
    using namespace tt::tt_metal::distributed;
    assert(plan && plan->initialized);

    const size_t total = static_cast<size_t>(plan->num_a_tiles) * kTileElems;

    // De-interleave host scratch.
    deinterleave_to_planes(a_padded, total, plan->a_r_host, plan->a_i_host);

    auto md = plan->md;
    MeshCommandQueue& cq = md->mesh_command_queue();

    WriteShard(cq, plan->a_r_buf, plan->a_r_host, MeshCoordinate(0, 0), false);
    WriteShard(cq, plan->a_i_buf, plan->a_i_host, MeshCoordinate(0, 0), false);
    EnqueueMeshWorkload(cq, plan->workload, false);
    ReadShard (cq, plan->out_r_host, plan->out_r_buf, MeshCoordinate(0, 0), true);
    ReadShard (cq, plan->out_i_host, plan->out_i_buf, MeshCoordinate(0, 0), true);

    interleave_from_planes(out_padded, total, plan->out_r_host, plan->out_i_host);
}

// ───────────────────────── chirp buffer cache ──────────────────────────────
//
// Convenience: upload a length-M chirp tensor (zero-padded from a length-N
// `w[n]` host vector) to device DRAM and return the (real, imag) MeshBuffers.
// Caller is expected to cache these inside `BluesteinPlan` so the chirp is
// uploaded once per N.

inline std::pair<std::shared_ptr<MeshBuffer>, std::shared_ptr<MeshBuffer>>
upload_chirp_padded(
    std::shared_ptr<MeshDevice> md,
    const Complex*              w,
    uint32_t                    N_active,
    uint32_t                    M_total)
{
    using namespace tt::tt_metal::distributed;
    assert(M_total >= N_active);
    assert(M_total % kTileElems == 0u);

    const uint32_t num_tiles = M_total / kTileElems;
    const uint32_t bytes     = num_tiles * kTileSizeFp32;

    auto r_buf = make_mesh_buf(md, bytes, kTileSizeFp32);
    auto i_buf = make_mesh_buf(md, bytes, kTileSizeFp32);

    std::vector<float> r_host(num_tiles * kTileElems, 0.0f);
    std::vector<float> i_host(num_tiles * kTileElems, 0.0f);
    for (uint32_t n = 0; n < N_active; ++n) {
        r_host[n] = w[n].real();
        i_host[n] = w[n].imag();
    }

    MeshCommandQueue& cq = md->mesh_command_queue();
    WriteShard(cq, r_buf, r_host, MeshCoordinate(0, 0), false);
    WriteShard(cq, i_buf, i_host, MeshCoordinate(0, 0), true);

    return {r_buf, i_buf};
}

// ───────────────────────── geometric eligibility ───────────────────────────
//
// Return true if (count, M) is a shape we can dispatch to the device kernel
// at all. Currently we require M >= one tile (1024 elements) so that the
// chirp is at least one tile, and num_a_tiles >= 1.
inline bool eligible(uint32_t count, uint32_t M) {
    return count >= 1u
        && M    >= kTileElems
        && (M % kTileElems) == 0u
        && (static_cast<uint64_t>(count) * M) >= kTileElems;
}

// ───────────────────────── Bluestein chirp pre/post mul ────────────────────
//
// Device-side replacements for the host loops in universal_host.hpp.
// Requires BluesteinPlan::chirp_dev_r/i and a cached ComplexMulPlan whose
// num_a_tiles matches (count * M) / kTileElems.

inline void device_chirp_premul(
    std::shared_ptr<MeshDevice>           md,
    const std::shared_ptr<ComplexMulPlan>& cm,
    uint32_t                              count,
    uint32_t                              N,
    uint32_t                              M,
    const std::vector<Complex>&           in,
    std::vector<Complex>&                 out_padded_M)
{
    (void)md;
    assert(cm && cm->initialized);
    assert(in.size() == static_cast<size_t>(count) * N);
    assert(out_padded_M.size() == static_cast<size_t>(count) * M);
    assert(cm->a_padded.size() == static_cast<size_t>(count) * M);

    for (uint32_t r = 0; r < count; ++r) {
        const Complex* src = in.data() + static_cast<size_t>(r) * N;
        Complex*       dst = cm->a_padded.data() + static_cast<size_t>(r) * M;
        std::copy(src, src + N, dst);
        // [N, M) stays zero from plan-build memset / prior calls.
    }

    run_complex_mul(cm, cm->a_padded.data(), out_padded_M.data());
}

inline void device_chirp_postmul(
    std::shared_ptr<MeshDevice>           md,
    const std::shared_ptr<ComplexMulPlan>& cm,
    uint32_t                              count,
    uint32_t                              N,
    uint32_t                              M,
    const std::vector<Complex>&           in_M,
    std::vector<Complex>&                 out_N)
{
    (void)md;
    assert(cm && cm->initialized);
    assert(in_M.size() == static_cast<size_t>(count) * M);
    assert(cm->out_padded_M.size() == static_cast<size_t>(count) * M);

    run_complex_mul(cm, in_M.data(), cm->out_padded_M.data());

    out_N.resize(static_cast<size_t>(count) * N);
    for (uint32_t r = 0; r < count; ++r) {
        const Complex* src = cm->out_padded_M.data() + static_cast<size_t>(r) * M;
        Complex*       dst = out_N.data() + static_cast<size_t>(r) * N;
        std::copy(src, src + N, dst);
    }
}

}  // namespace fft_complex_mul
