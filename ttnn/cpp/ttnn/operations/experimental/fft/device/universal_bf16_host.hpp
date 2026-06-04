// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_universal_bf16_host.cpp — TRUE-bf16 variant of fft_universal.
//

#pragma once

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/tilize_utils.hpp"

#include "stockham_host.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fft_universal_bf16 {

using Complex = std::complex<float>;
using tt::tt_metal::distributed::MeshDevice;

constexpr uint32_t kPackedMaxN = 32u;

// ─── Small helpers ───────────────────────────────────────────────────────────
inline bool is_pow2(uint32_t n) { return n != 0u && (n & (n - 1u)) == 0u; }

// Smallest pow2 ≥ n, with n ≥ 1.
inline uint32_t next_pow2(uint32_t n) {
    if (n <= 1u) return 1u;
    uint32_t p = 1u;
    while (p < n) p <<= 1u;
    return p;
}

// Deterministic primality test — trial division up to sqrt(N). Fine for the
// sizes we care about (N ≤ 10^6): at most ~1000 modulo ops.
inline bool is_prime(uint32_t n) {
    if (n < 2u) return false;
    if (n == 2u || n == 3u) return true;
    if ((n & 1u) == 0u) return false;
    if (n % 3u == 0u) return false;
    for (uint32_t i = 5u; i * i <= n; i += 6u) {
        if (n % i == 0u) return false;
        if (n % (i + 2u) == 0u) return false;
    }
    return true;
}

inline uint32_t largest_divisor_le_32(uint32_t N) {
    uint32_t best = 0u;
    for (uint32_t d = 32u; d >= 2u; --d) {
        if (N % d == 0u) { best = d; break; }
    }
    return best;
}

// Forward decl — the algorithm helpers below recurse via fft() for
// sub-FFTs. Definition is at the bottom of the file.
inline std::vector<Complex> fft(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<Complex>&  signal);

// ─── bf16 conversion (round-to-nearest-even, IEEE compliant) ────────────────
inline uint16_t fp32_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u) {
        // NaN: preserve payload bit so result is still NaN after truncation.
        return static_cast<uint16_t>((bits >> 16) | 0x40u);
    }
    const uint32_t lsb  = (bits >> 16) & 1u;
    const uint32_t bias = 0x7FFFu + lsb;
    bits += bias;
    return static_cast<uint16_t>(bits >> 16);
}

inline float bf16_to_fp32(uint16_t b) {
    const uint32_t bits = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// Convert a vector of fp32 values to bf16 (stored as uint16_t). No layout
// change — this is purely elementwise.
inline std::vector<uint16_t> fp32_to_bf16_vec(const std::vector<float>& src) {
    std::vector<uint16_t> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) dst[i] = fp32_to_bf16(src[i]);
    return dst;
}

inline std::vector<float> bf16_to_fp32_vec(const std::vector<uint16_t>& src) {
    std::vector<float> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) dst[i] = bf16_to_fp32(src[i]);
    return dst;
}

struct PackedDFTBf16Plan {
    uint32_t N              = 0;
    uint32_t count          = 0;
    uint32_t num_tiles      = 0;
    uint32_t num_cores      = 0;
    uint32_t tiles_per_core = 0;
    uint32_t grid_cols      = 0;
    uint32_t grid_rows      = 0;

    std::shared_ptr<MeshDevice> md;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> in_r_buf,  in_i_buf;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> out_r_buf, out_i_buf;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> tw_r_buf, tw_i_buf, tw_i_neg_buf;
    tt::tt_metal::distributed::MeshWorkload workload;

    // Host scratch. Row-major fp32 mirrors are used for packing / twiddle
    // prep; tilized fp32 is the layout fed to tilize_nfaces; uint16_t
    // scratch holds the bf16 tiles we actually ship to DRAM.
    std::vector<float>    in_r_rm,   in_i_rm;      // row-major host layout (fp32)
    std::vector<float>    in_r_til_f32, in_i_til_f32;  // tilized (fp32 mirror)
    std::vector<uint16_t> in_r_til,  in_i_til;     // bf16 tile bytes → DRAM
    std::vector<uint16_t> out_r_til, out_i_til;    // bf16 tile bytes ← DRAM
    std::vector<float>    out_r_rm,  out_i_rm;     // row-major host (fp32 after convert)

    bool initialized = false;
};

inline uint32_t bf16_tile_bytes() {
    return 32u * 32u * sizeof(uint16_t);   // 2048
}

// Build the 32×32 complex twiddle matrix T[n, k] = exp(-2πi · k · n / N)
// in ROW-MAJOR layout (cos in tr_rm, sin in ti_rm). Entries outside
// [0, N)² stay zero so padding rows/cols contribute nothing.
inline std::pair<std::vector<float>, std::vector<float>>
packed_dft_twiddle_rm(uint32_t N) {
    std::vector<float> tr(32u * 32u, 0.0f), ti(32u * 32u, 0.0f);
    const double tau_over_N = -2.0 * M_PI / static_cast<double>(N);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t k = 0; k < N; ++k) {
            const double a = tau_over_N * static_cast<double>(n) * static_cast<double>(k);
            tr[n * 32u + k] = static_cast<float>(std::cos(a));
            ti[n * 32u + k] = static_cast<float>(std::sin(a));
        }
    }
    return {std::move(tr), std::move(ti)};
}

inline std::shared_ptr<PackedDFTBf16Plan> make_packed_dft_bf16_plan(
    std::shared_ptr<MeshDevice> md, uint32_t N, uint32_t count)
{
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;

    auto pp = std::make_shared<PackedDFTBf16Plan>();
    pp->md    = md;
    pp->N     = N;
    pp->count = count;

    assert(N >= 2u && N <= kPackedMaxN);
    assert(count >= 1u);

    constexpr uint32_t kRowsPerTile = 32u;
    const uint32_t raw_num_tiles = (count + kRowsPerTile - 1u) / kRowsPerTile;

    const auto     dev_grid  = md->compute_with_storage_grid_size();
    const uint32_t grid_x    = static_cast<uint32_t>(dev_grid.x);
    const uint32_t max_cores = grid_x * static_cast<uint32_t>(dev_grid.y);

    uint32_t num_cores;
    if (raw_num_tiles < grid_x) {
        num_cores = raw_num_tiles;
    } else {
        num_cores = ((raw_num_tiles + grid_x - 1u) / grid_x) * grid_x;
        if (num_cores > max_cores) num_cores = max_cores;
    }
    const uint32_t tiles_per_core = (raw_num_tiles + num_cores - 1u) / num_cores;
    const uint32_t num_tiles      = num_cores * tiles_per_core;

    pp->num_cores      = num_cores;
    pp->tiles_per_core = tiles_per_core;
    pp->num_tiles      = num_tiles;
    std::tie(pp->grid_cols, pp->grid_rows) = fft_stockham::pick_batch_grid(num_cores, grid_x);

    // (dev-time stdout printf removed — plan summary is internal detail and
    // shouldn't leak into user output. Re-enable temporarily under a local
    // #if 0 guard if you need to debug a new dispatch shape.)

    MeshCommandQueue& cq = md->mesh_command_queue();

    // ── DRAM buffers (bf16 tile = 2048 B) ───────────────────────────────
    const uint32_t ts_bf16  = bf16_tile_bytes();
    const uint32_t io_bytes = num_tiles * ts_bf16;
    pp->in_r_buf  = fft_stockham::make_mesh_buf(md, io_bytes, ts_bf16);
    pp->in_i_buf  = fft_stockham::make_mesh_buf(md, io_bytes, ts_bf16);
    pp->out_r_buf = fft_stockham::make_mesh_buf(md, io_bytes, ts_bf16);
    pp->out_i_buf = fft_stockham::make_mesh_buf(md, io_bytes, ts_bf16);

    // Single-tile twiddle buffers (T_R, T_I, T_I_neg).
    pp->tw_r_buf     = fft_stockham::make_mesh_buf(md, ts_bf16, ts_bf16);
    pp->tw_i_buf     = fft_stockham::make_mesh_buf(md, ts_bf16, ts_bf16);
    pp->tw_i_neg_buf = fft_stockham::make_mesh_buf(md, ts_bf16, ts_bf16);

    // Build twiddles in fp32, tilize, convert to bf16, ship to DRAM.
    // The fp32 → bf16 rounding on the twiddles happens once per plan;
    // subsequent calls reuse the already-tilized bf16 tiles.
    auto [tr_rm, ti_rm] = packed_dft_twiddle_rm(N);
    std::vector<float> ti_neg_rm(ti_rm.size());
    for (size_t i = 0; i < ti_rm.size(); ++i) ti_neg_rm[i] = -ti_rm[i];

    // NOTE: WriteShard takes std::vector<DType>& (non-const reference), so
    // these tile scratch vectors must be non-const locals. Keeping them
    // const here yields a template-deduction failure against distributed.hpp.
    std::vector<float> tr_til_f32     = tilize_nfaces(tr_rm,     32u, 32u);
    std::vector<float> ti_til_f32     = tilize_nfaces(ti_rm,     32u, 32u);
    std::vector<float> ti_neg_til_f32 = tilize_nfaces(ti_neg_rm, 32u, 32u);

    std::vector<uint16_t> tr_til     = fp32_to_bf16_vec(tr_til_f32);
    std::vector<uint16_t> ti_til     = fp32_to_bf16_vec(ti_til_f32);
    std::vector<uint16_t> ti_neg_til = fp32_to_bf16_vec(ti_neg_til_f32);

    WriteShard(cq, pp->tw_r_buf,     tr_til,     MeshCoordinate(0, 0), false);
    WriteShard(cq, pp->tw_i_buf,     ti_til,     MeshCoordinate(0, 0), false);
    WriteShard(cq, pp->tw_i_neg_buf, ti_neg_til, MeshCoordinate(0, 0), false);

    // ── Host scratch ────────────────────────────────────────────────────
    const size_t rm_floats  = static_cast<size_t>(num_tiles) * 32u * 32u;
    const size_t til_elems  = static_cast<size_t>(num_tiles) * fft_stockham::kTileElems;
    pp->in_r_rm .assign(rm_floats, 0.0f);
    pp->in_i_rm .assign(rm_floats, 0.0f);
    pp->in_r_til_f32.assign(til_elems, 0.0f);
    pp->in_i_til_f32.assign(til_elems, 0.0f);
    pp->in_r_til .assign(til_elems, 0u);
    pp->in_i_til .assign(til_elems, 0u);
    pp->out_r_til.assign(til_elems, 0u);
    pp->out_i_til.assign(til_elems, 0u);
    pp->out_r_rm .assign(rm_floats, 0.0f);
    pp->out_i_rm .assign(rm_floats, 0.0f);

    // ── Program ─────────────────────────────────────────────────────────
    Program prog = CreateProgram();

    const CoreCoord first{0, 0};
    const CoreCoord last{pp->grid_cols - 1, pp->grid_rows - 1};
    const CoreRange cr(first, last);

    // CBs: CB_A / CB_B depth 4 so the reader can queue all 4 matmul pairs
    // upfront. CB_OUT_R / CB_OUT_I depth 2, ordinary double-buffer.
    constexpr uint32_t kCbCount = 4u;
    constexpr uint32_t kCbTiles[kCbCount] = {4u, 4u, 2u, 2u};
    for (uint32_t id = 0; id < kCbCount; ++id) {
        CircularBufferConfig c(
            kCbTiles[id] * ts_bf16,
            {{id, tt::DataFormat::Float16_b}});
        c.set_page_size(id, ts_bf16);
        CreateCircularBuffer(prog, cr, c);
    }

    auto rk = CreateKernel(
        prog,
        "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/packed_dft_bf16_reader.cpp",
        cr,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc       = NOC::RISCV_0_default});

    auto wk = CreateKernel(
        prog,
        "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/packed_dft_bf16_writer.cpp",
        cr,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc       = NOC::RISCV_1_default});

    auto ck = CreateKernel(
        prog,
        "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/compute/packed_dft_bf16_compute.cpp",
        cr,
        ComputeConfig{
            .math_fidelity    = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .compile_args     = {pp->tiles_per_core}});
    (void)ck;

    for (uint32_t c = 0; c < pp->num_cores; ++c) {
        const CoreCoord logical = fft_stockham::batch_logical_core(c, pp->grid_cols);
        const uint32_t  base    = c * pp->tiles_per_core;

        SetRuntimeArgs(prog, rk, logical, {
            fft_stockham::buf_addr(pp->in_r_buf),
            fft_stockham::buf_addr(pp->in_i_buf),
            fft_stockham::buf_addr(pp->tw_r_buf),
            fft_stockham::buf_addr(pp->tw_i_buf),
            fft_stockham::buf_addr(pp->tw_i_neg_buf),
            base,
            pp->tiles_per_core,
        });

        SetRuntimeArgs(prog, wk, logical, {
            fft_stockham::buf_addr(pp->out_r_buf),
            fft_stockham::buf_addr(pp->out_i_buf),
            base,
            pp->tiles_per_core,
        });
    }

    pp->workload.add_program(
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 0)),
        std::move(prog));
    pp->initialized = true;
    return pp;
}

namespace detail_packed {
inline std::unordered_map<uint64_t, std::shared_ptr<PackedDFTBf16Plan>>&
packed_dft_bf16_cache() {
    static std::unordered_map<uint64_t, std::shared_ptr<PackedDFTBf16Plan>> c;
    return c;
}
inline uint64_t packed_dft_bf16_key(MeshDevice* md, uint32_t N, uint32_t count) {
    return reinterpret_cast<uint64_t>(md)
         ^ (uint64_t{N}     * 0x9E3779B97F4A7C15ull)
         ^ (uint64_t{count} * 0xBF58476D1CE4E5B9ull);
}
}  // namespace detail_packed

inline std::shared_ptr<PackedDFTBf16Plan> get_cached_packed_dft_bf16_plan(
    std::shared_ptr<MeshDevice> md, uint32_t N, uint32_t count)
{
    const uint64_t key = detail_packed::packed_dft_bf16_key(md.get(), N, count);
    auto& cache = detail_packed::packed_dft_bf16_cache();
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    auto pp = make_packed_dft_bf16_plan(md, N, count);
    cache.emplace(key, pp);
    return pp;
}

inline void packed_direct_dft_bf16_batched(
    std::shared_ptr<MeshDevice>   md,
    uint32_t                      N,
    uint32_t                      count,
    const std::vector<Complex>&   in_natural,
    std::vector<Complex>&         out_natural)
{
    using namespace tt::tt_metal::distributed;
    assert(in_natural.size() == static_cast<size_t>(count) * N);

    auto plan = get_cached_packed_dft_bf16_plan(md, N, count);
    constexpr uint32_t kRowsPerTile = 32u;

    std::vector<float>&    in_r_rm      = plan->in_r_rm;
    std::vector<float>&    in_i_rm      = plan->in_i_rm;
    std::vector<float>&    in_r_til_f32 = plan->in_r_til_f32;
    std::vector<float>&    in_i_til_f32 = plan->in_i_til_f32;
    std::vector<uint16_t>& in_r_til     = plan->in_r_til;
    std::vector<uint16_t>& in_i_til     = plan->in_i_til;
    std::vector<uint16_t>& out_r_til    = plan->out_r_til;
    std::vector<uint16_t>& out_i_til    = plan->out_i_til;

    // Pack natural-order fp32 input into row-major (32 * num_tiles) × 32.
    // Slot [r * 32 + k] for k ∈ [0, N) is sub-FFT r's sample k. Slots
    // [N, 32) stay zero. Rows r ∈ [count, 32 * num_tiles) stay zero.
    for (uint32_t r = 0; r < count; ++r) {
        const Complex* src = in_natural.data() + static_cast<size_t>(r) * N;
        float* tr = in_r_rm.data() + static_cast<size_t>(r) * 32u;
        float* ti = in_i_rm.data() + static_cast<size_t>(r) * 32u;
        for (uint32_t k = 0; k < N; ++k) {
            tr[k] = src[k].real();
            ti[k] = src[k].imag();
        }
    }

    // Row-major → tilized (still fp32) → bf16 tile bytes.
    const uint32_t total_rows = plan->num_tiles * kRowsPerTile;
    in_r_til_f32 = tilize_nfaces(in_r_rm, total_rows, 32u);
    in_i_til_f32 = tilize_nfaces(in_i_rm, total_rows, 32u);
    for (size_t i = 0; i < in_r_til_f32.size(); ++i) {
        in_r_til[i] = fp32_to_bf16(in_r_til_f32[i]);
        in_i_til[i] = fp32_to_bf16(in_i_til_f32[i]);
    }

    MeshCommandQueue& cq = plan->md->mesh_command_queue();
    WriteShard(cq, plan->in_r_buf, in_r_til, MeshCoordinate(0, 0), false);
    WriteShard(cq, plan->in_i_buf, in_i_til, MeshCoordinate(0, 0), false);

    EnqueueMeshWorkload(cq, plan->workload, false);

    ReadShard(cq, out_r_til, plan->out_r_buf, MeshCoordinate(0, 0), true);
    ReadShard(cq, out_i_til, plan->out_i_buf, MeshCoordinate(0, 0), true);

    // bf16 → fp32 → untilize → per-row unpack.
    const std::vector<float> out_r_til_f32 = bf16_to_fp32_vec(out_r_til);
    const std::vector<float> out_i_til_f32 = bf16_to_fp32_vec(out_i_til);
    std::vector<float>& out_r_rm = plan->out_r_rm;
    std::vector<float>& out_i_rm = plan->out_i_rm;
    out_r_rm = untilize_nfaces(out_r_til_f32, total_rows, 32u);
    out_i_rm = untilize_nfaces(out_i_til_f32, total_rows, 32u);

    out_natural.resize(static_cast<size_t>(count) * N);
    for (uint32_t r = 0; r < count; ++r) {
        const float* tr = out_r_rm.data() + static_cast<size_t>(r) * 32u;
        const float* ti = out_i_rm.data() + static_cast<size_t>(r) * 32u;
        Complex*     dst = out_natural.data() + static_cast<size_t>(r) * N;
        for (uint32_t k = 0; k < N; ++k) dst[k] = {tr[k], ti[k]};
    }
}

inline std::vector<Complex>& two_level_twiddle_cache_entry(
    uint32_t N, uint32_t N1, uint32_t N2)
{
    static std::unordered_map<uint64_t, std::vector<Complex>> cache;
    const uint64_t key = (uint64_t{N}  * 0x9E3779B97F4A7C15ull)
                       ^ (uint64_t{N1} * 0xBF58476D1CE4E5B9ull)
                       ^ (uint64_t{N2} * 0x94D049BB133111EBull);
    return cache[key];
}

inline const Complex* get_two_level_twiddle_cached(
    uint32_t N, uint32_t N1, uint32_t N2)
{
    auto& tab = two_level_twiddle_cache_entry(N, N1, N2);
    if (tab.size() == static_cast<size_t>(N1) * N2) return tab.data();

    tab.assign(static_cast<size_t>(N1) * N2, Complex{0.0f, 0.0f});
    const double two_pi_over_N = -2.0 * M_PI / static_cast<double>(N);
    for (uint32_t n1 = 0; n1 < N1; ++n1) {
        for (uint32_t k2 = 0; k2 < N2; ++k2) {
            const double angle =
                two_pi_over_N * static_cast<double>(n1) * static_cast<double>(k2);
            tab[static_cast<size_t>(n1) * N2 + k2] = Complex{
                static_cast<float>(std::cos(angle)),
                static_cast<float>(std::sin(angle))
            };
        }
    }
    return tab.data();
}

inline void batched_two_level_fft_bf16(
    std::shared_ptr<MeshDevice>  md,
    uint32_t                     N,
    uint32_t                     N1,
    uint32_t                     batch,
    const std::vector<Complex>&  in_natural,
    std::vector<Complex>&        out_natural)
{
    assert(N1 >= 2u && N1 <= kPackedMaxN);
    assert(N % N1 == 0u);
    const uint32_t N2 = N / N1;
    assert(N2 >= 2u && N2 <= kPackedMaxN);
    assert(in_natural.size() == static_cast<size_t>(batch) * N);

    // ── Step 1: per-batch reshape into (batch · N1) sub-FFTs of length N2.
    // Layout: pass1_in[ (b·N1 + n1) · N2 + n2 ] = in_natural[ b·N + n1 + N1·n2 ]
    std::vector<Complex> pass1_in(static_cast<size_t>(batch) * N);
    for (uint32_t b = 0; b < batch; ++b) {
        const Complex* src = in_natural.data() + static_cast<size_t>(b) * N;
        for (uint32_t n1 = 0; n1 < N1; ++n1) {
            Complex* dst = pass1_in.data()
                         + (static_cast<size_t>(b) * N1 + n1) * N2;
            for (uint32_t n2 = 0; n2 < N2; ++n2) {
                dst[n2] = src[n1 + N1 * n2];
            }
        }
    }

    // ── Step 2: ONE big Pass-1 dispatch — batch·N1 length-N2 FFTs.
    std::vector<Complex> A_prime;
    packed_direct_dft_bf16_batched(md, /*N=*/N2, /*count=*/batch * N1,
                                   pass1_in, A_prime);
    // A_prime[(b·N1 + n1)·N2 + k2] = result of sub-FFT (b, n1) at bin k2.

    // ── Step 3: per-batch host twiddle + transpose (cached twiddles).
    // Twiddle table T[n1, k2] = exp(-2πi · n1 · k2 / N) is the same for
    // every batch element; cache it once via the (N, N1, N2) key.
    const Complex* __restrict__ twid =
        get_two_level_twiddle_cached(N, N1, N2);
    std::vector<Complex> pass2_in(static_cast<size_t>(batch) * N);
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t n1 = 0; n1 < N1; ++n1) {
            const Complex* tw_row = twid + static_cast<size_t>(n1) * N2;
            const Complex* a_row  = A_prime.data()
                                  + (static_cast<size_t>(b) * N1 + n1) * N2;
            // Per-batch Pass-2 input: batch b's tile starts at b·N, then
            // sub-FFT k2 (length N1) at offset k2·N1.
            Complex* p2_b = pass2_in.data() + static_cast<size_t>(b) * N;
            for (uint32_t k2 = 0; k2 < N2; ++k2) {
                p2_b[k2 * N1 + n1] = a_row[k2] * tw_row[k2];
            }
        }
    }

    // ── Step 4: ONE big Pass-2 dispatch — batch·N2 length-N1 FFTs.
    std::vector<Complex> D;
    packed_direct_dft_bf16_batched(md, /*N=*/N1, /*count=*/batch * N2,
                                   pass2_in, D);
    // D[(b·N2 + k2)·N1 + k1] = final bin (k2, k1) of batch b.

    // ── Step 5: per-batch output permutation. natural-order index k for
    // batch b is k = k2 + N2·k1 (matches the non-batched two_level path).
    out_natural.assign(static_cast<size_t>(batch) * N, Complex{0.0f, 0.0f});
    for (uint32_t b = 0; b < batch; ++b) {
        const Complex* d_b   = D.data()           + static_cast<size_t>(b) * N;
        Complex*       out_b = out_natural.data() + static_cast<size_t>(b) * N;
        for (uint32_t k2 = 0; k2 < N2; ++k2) {
            const Complex* d_row = d_b + static_cast<size_t>(k2) * N1;
            for (uint32_t k1 = 0; k1 < N1; ++k1) {
                out_b[k2 + N2 * k1] = d_row[k1];
            }
        }
    }
}

inline void two_level_fft_bf16(
    std::shared_ptr<MeshDevice>  md,
    uint32_t                     N,
    uint32_t                     N1,
    const std::vector<Complex>&  in_natural,
    std::vector<Complex>&        out_natural)
{
    assert(N1 >= 2u && N1 <= kPackedMaxN);
    assert(N % N1 == 0u);
    const uint32_t N2 = N / N1;
    assert(in_natural.size() == N);

    // Step 1: reshape x into (N1 sub-FFTs) × (N2 samples each), with stride-N1
    // slicing. pass1_in sub-FFT r (= n1) = [ x[r], x[r+N1], x[r+2N1], ... ].
    std::vector<Complex> pass1_in(N);
    for (uint32_t n1 = 0; n1 < N1; ++n1) {
        for (uint32_t n2 = 0; n2 < N2; ++n2) {
            pass1_in[n1 * N2 + n2] = in_natural[n1 + N1 * n2];
        }
    }

    std::vector<Complex> A_prime(N);
    if (N2 <= kPackedMaxN) {
        packed_direct_dft_bf16_batched(md, /*N=*/N2, /*count=*/N1, pass1_in, A_prime);
    } else if (is_pow2(N2) && (N2 % kPackedMaxN == 0u)
                           && (N2 / kPackedMaxN) <= kPackedMaxN) {
        // N2 = 32 · k with k ≤ 32 → batched two-level for all N1 sub-FFTs at once.
        // (in_natural shape for the batched call is exactly pass1_in: N1 sub-
        // FFTs of length N2, contiguous. Here `batch` plays the role of N1.)
        batched_two_level_fft_bf16(md, /*N=*/N2, /*N1=*/kPackedMaxN,
                                   /*batch=*/N1, pass1_in, A_prime);
    } else {
        for (uint32_t r = 0; r < N1; ++r) {
            std::vector<Complex> sub_in(
                pass1_in.begin() + static_cast<ptrdiff_t>(r) * N2,
                pass1_in.begin() + static_cast<ptrdiff_t>(r + 1u) * N2);
            std::vector<Complex> sub_out = fft(md, sub_in);
            std::copy(sub_out.begin(), sub_out.end(),
                      A_prime.begin() + static_cast<ptrdiff_t>(r) * N2);
        }
    }

    const Complex* __restrict__ twid =
        get_two_level_twiddle_cached(N, N1, N2);
    std::vector<Complex> pass2_in(N);
    for (uint32_t n1 = 0; n1 < N1; ++n1) {
        const Complex* __restrict__ tw_row = twid + static_cast<size_t>(n1) * N2;
        const Complex* __restrict__ a_row  = A_prime.data() + static_cast<size_t>(n1) * N2;
        for (uint32_t k2 = 0; k2 < N2; ++k2) {
            pass2_in[k2 * N1 + n1] = a_row[k2] * tw_row[k2];
        }
    }

    // Step 4: Pass-2 — N2 sibling length-N1 FFTs. Always ≤ 32 so the
    // Phase-1 packed direct-DFT kernel handles it in one dispatch.
    std::vector<Complex> D;
    packed_direct_dft_bf16_batched(md, /*N=*/N1, /*count=*/N2, pass2_in, D);

    // Step 5: output permutation. D[k2, k1] (= D[k2·N1 + k1] linearised)
    // lands at X[k2 + N2·k1].
    out_natural.assign(N, Complex{0.0f, 0.0f});
    for (uint32_t k2 = 0; k2 < N2; ++k2) {
        for (uint32_t k1 = 0; k1 < N1; ++k1) {
            out_natural[k2 + N2 * k1] = D[k2 * N1 + k1];
        }
    }
}

inline void pow2_fft_bf16(
    std::shared_ptr<MeshDevice>  md,
    uint32_t                     N,
    const std::vector<Complex>&  in_natural,
    std::vector<Complex>&        out_natural)
{
    assert(is_pow2(N) && N > kPackedMaxN);
    two_level_fft_bf16(md, N, /*N1=*/32u, in_natural, out_natural);
}

inline void mixed_radix_fft_bf16(
    std::shared_ptr<MeshDevice>  md,
    uint32_t                     N,
    const std::vector<Complex>&  in_natural,
    std::vector<Complex>&        out_natural)
{
    const uint32_t N1 = largest_divisor_le_32(N);
    assert(N1 >= 2u && "mixed_radix expects a divisor ≤ 32 — caller must check");
    two_level_fft_bf16(md, N, N1, in_natural, out_natural);
}

inline void bluestein_fft_bf16(
    std::shared_ptr<MeshDevice>  md,
    uint32_t                     N,
    const std::vector<Complex>&  in_natural,
    std::vector<Complex>&        out_natural)
{
    assert(N > kPackedMaxN);
    assert(in_natural.size() == N);

    const uint32_t M = next_pow2(2u * N - 1u);

    // Build chirp c[n] = exp(-iπ n² / N) for n ∈ [0, N). Use doubles for
    // the exponent because n² grows quadratically and we want the mod-2π
    // reduction to be exact in the phase domain.
    std::vector<Complex> c(N);
    {
        const double pi_over_N = M_PI / static_cast<double>(N);
        for (uint32_t n = 0; n < N; ++n) {
            const double n2 = static_cast<double>(n) * static_cast<double>(n);
            const double angle = -pi_over_N * n2;
            c[n] = {static_cast<float>(std::cos(angle)),
                    static_cast<float>(std::sin(angle))};
        }
    }

    // a[n] = x[n] · c[n], zero-padded to length M.
    std::vector<Complex> a(M, Complex{0.0f, 0.0f});
    for (uint32_t n = 0; n < N; ++n) {
        a[n] = in_natural[n] * c[n];
    }

    // b is the conjugate chirp arranged for a linear (not cyclic)
    // convolution with a. b[0] = conj(c[0]); b[n] = b[M-n] = conj(c[n])
    // for n ∈ [1, N). Everything else is zero.
    std::vector<Complex> b(M, Complex{0.0f, 0.0f});
    b[0] = std::conj(c[0]);
    for (uint32_t n = 1; n < N; ++n) {
        const Complex cc = std::conj(c[n]);
        b[n]     = cc;
        b[M - n] = cc;
    }

    // Three length-M bf16 FFTs (A, B, and the inner FFT of the IFFT).
    const std::vector<Complex> A = fft(md, a);
    const std::vector<Complex> B = fft(md, b);

    // P = A * B (pointwise, host fp32).
    std::vector<Complex> P_conj(M);
    for (uint32_t k = 0; k < M; ++k) {
        const Complex p = A[k] * B[k];
        P_conj[k] = std::conj(p);   // pre-conjugate for the inverse trick
    }

    // IFFT_M(P) = conj(FFT_M(conj(P))) / M.
    const std::vector<Complex> P_fft = fft(md, P_conj);
    const float inv_M = 1.0f / static_cast<float>(M);
    std::vector<Complex> p(M);
    for (uint32_t k = 0; k < M; ++k) {
        p[k] = std::conj(P_fft[k]) * inv_M;
    }

    // X[k] = c[k] · p[k], k ∈ [0, N).
    out_natural.assign(N, Complex{0.0f, 0.0f});
    for (uint32_t k = 0; k < N; ++k) {
        out_natural[k] = c[k] * p[k];
    }
}

inline std::vector<Complex> fft(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<Complex>&  signal)
{
    const uint32_t N = static_cast<uint32_t>(signal.size());
    if (N == 1u) return signal;
    assert(N >= 1u && "FFT requires N >= 1");

    // Phase 1 — packed direct-DFT for every small N.
    if (N >= 2u && N <= kPackedMaxN) {
        std::vector<Complex> out;
        packed_direct_dft_bf16_batched(md, N, /*count=*/1u, signal, out);
        return out;
    }

    // Phase 2b — pow2 recursion.
    if (is_pow2(N)) {
        std::vector<Complex> out;
        pow2_fft_bf16(md, N, signal, out);
        return out;
    }

    // Phase 2c — composites with at least one divisor ≤ 32 go mixed-radix.
    if (largest_divisor_le_32(N) >= 2u) {
        std::vector<Complex> out;
        mixed_radix_fft_bf16(md, N, signal, out);
        return out;
    }

    // Phase 2c — primes, and composites whose prime factors are all > 32,
    // go Bluestein.
    std::vector<Complex> out;
    bluestein_fft_bf16(md, N, signal, out);
    return out;
}

// Convenience overload for real input.
inline std::vector<Complex> fft(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<float>&    real_signal)
{
    std::vector<Complex> cx(real_signal.size());
    for (size_t i = 0; i < real_signal.size(); ++i) {
        cx[i] = Complex(real_signal[i], 0.0f);
    }
    return fft(md, cx);
}

inline std::vector<Complex> ifft(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<Complex>&  spectrum)
{
    const uint32_t N = static_cast<uint32_t>(spectrum.size());
    if (N == 1u) return spectrum;
    assert(N >= 1u && "IFFT requires N >= 1");

    std::vector<Complex> conj_in(N);
    for (uint32_t k = 0; k < N; ++k) conj_in[k] = std::conj(spectrum[k]);

    const std::vector<Complex> fwd = fft(md, conj_in);

    const float inv_N = 1.0f / static_cast<float>(N);
    std::vector<Complex> out(N);
    for (uint32_t k = 0; k < N; ++k) out[k] = std::conj(fwd[k]) * inv_N;
    return out;
}

// Convenience overload for real-valued spectrum input.
inline std::vector<Complex> ifft(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<float>&    real_spectrum)
{
    std::vector<Complex> cx(real_spectrum.size());
    for (size_t i = 0; i < real_spectrum.size(); ++i) {
        cx[i] = Complex(real_spectrum[i], 0.0f);
    }
    return ifft(md, cx);
}

}  // namespace fft_universal_bf16
