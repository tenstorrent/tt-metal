// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_stockham_host.cpp — Multi-pass Stockham (six-step / Bailey 4-step) FFT
//                        orchestrator that lifts our radix-2 single-shot FFT

#pragma once

#include "tt-metalium/host_api.hpp"
#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/mesh_command_queue.hpp"
#include "tt-metalium/mesh_workload.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/circular_buffer_constants.h"   // NUM_CIRCULAR_BUFFERS (host max → 64)

#include "fft_inner_host.hpp"   // reuse the inner radix-2 kernel & plan cache

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>
#include <utility>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <memory>
#include <unordered_map>

// fft/fft_host.cpp does `using namespace tt::tt_metal::distributed;` at file
// scope, so MeshDevice and friends are visible here without further qualification.

namespace fft_stockham {

using Complex = std::complex<float>;
using fft_example::log2u;
using fft_example::bit_rev;
using fft_example::kTileElems;
using fft_example::kTileSizeFp32;
using fft_example::make_mesh_buf;
using fft_example::buf_addr;
using tt::tt_metal::distributed::MeshDevice;

// ── Sizing & factorisation ────────────────────────────────────────────────

// Maximum N a single inner radix-2 dispatch can handle.
constexpr uint32_t kInnerMaxN = 65536u;

// Power-of-two check.
inline bool is_pow2(uint32_t n) { return n != 0 && (n & (n - 1)) == 0; }

struct StockhamPlan {
    uint32_t N        = 0;
    uint32_t N1       = 0;     // outer (column-FFT) dimension
    uint32_t N2       = 0;     // inner (row-FFT)    dimension
    bool     stockham = false; // false => fall through to inner radix-2
};

// Choose a balanced factorisation N = N1 * N2 such that both halves fit in
// the inner radix-2 kernel. Strategy: pick N2 = sqrt(N) rounded to the next
// power of two; clamp to kInnerMaxN. This keeps both passes well L1-resident.
inline StockhamPlan plan(uint32_t N) {
    StockhamPlan p{};
    p.N = N;

    if (N <= kInnerMaxN) { p.stockham = false; p.N1 = N; p.N2 = 1; return p; }

    assert(is_pow2(N) && "Stockham path requires N to be a power of two.");

    // log2N is the total number of butterfly stages.
    const uint32_t log2N = log2u(N);

    // Split log2N as evenly as possible, then clamp each half to fit the
    // inner kernel (at most log2(kInnerMaxN) = 16 bits per pass).
    uint32_t log2N2 = log2N / 2;             // inner / row-FFT length
    uint32_t log2N1 = log2N - log2N2;        // outer / column-FFT length
    const uint32_t log2_inner_max = log2u(kInnerMaxN);
    if (log2N1 > log2_inner_max) {
        const uint32_t spill = log2N1 - log2_inner_max;
        log2N1 -= spill;
        log2N2 += spill;
    }
    if (log2N2 > log2_inner_max) {
        const uint32_t spill = log2N2 - log2_inner_max;
        log2N2 -= spill;
        log2N1 += spill;
    }

    p.N1 = 1u << log2N1;
    p.N2 = 1u << log2N2;
    p.stockham = true;

    assert(p.N1 <= kInnerMaxN && p.N2 <= kInnerMaxN);
    assert(static_cast<uint64_t>(p.N1) * static_cast<uint64_t>(p.N2) ==
           static_cast<uint64_t>(p.N));
    return p;
}

struct BatchFFTPlan {
    uint32_t sub_N          = 0;
    uint32_t log2_sub_N     = 0;
    uint32_t batch          = 0;     // total number of sub-FFTs
    uint32_t num_cores      = 0;     // <= 64
    uint32_t batch_per_core = 0;     // batch / num_cores  (must divide cleanly)
    uint32_t grid_cols      = 0;
    uint32_t grid_rows      = 0;

    std::shared_ptr<MeshDevice> md;
    std::shared_ptr<MeshBuffer> in_r_buf,  in_i_buf;
    std::shared_ptr<MeshBuffer> out_r_buf, out_i_buf;
    std::shared_ptr<MeshBuffer> tw_r_buf,  tw_i_buf;
    tt::tt_metal::distributed::MeshWorkload workload;

    std::vector<float> in_r_host, in_i_host;
    std::vector<float> out_r_host, out_i_host;

    bool initialized = false;
};

inline std::pair<uint32_t, uint32_t> pick_batch_grid(uint32_t num_cores, uint32_t grid_x) {
    // Search downward from grid_x for the largest divisor of num_cores
    // that is <= grid_x. Guaranteed to terminate at cols=1 in the worst case.
    uint32_t cols = (num_cores < grid_x) ? num_cores : grid_x;
    while (cols > 1 && num_cores % cols != 0) {
        --cols;
    }
    return {cols, num_cores / cols};
}

inline uint32_t max_cores_for_grid(uint32_t grid_x, uint32_t grid_y) {
    uint32_t best = 1;
    for (uint32_t p = 2; p <= grid_x * grid_y; p *= 2) {
        // Is there cols <= grid_x dividing p with p/cols <= grid_y?
        bool ok = false;
        for (uint32_t c = std::min(p, grid_x); c >= 1; --c) {
            if (p % c == 0 && p / c <= grid_y) { ok = true; break; }
        }
        if (ok) best = p;
    }
    return best;
}

inline tt::tt_metal::CoreCoord batch_logical_core(
    uint32_t c, uint32_t grid_cols)
{
    return tt::tt_metal::CoreCoord{c % grid_cols, c / grid_cols};
}

// LOG2_SUB_N tiles per side; tile s holds the stage-s twiddles for a
// single-tile (P=1) radix-2 sub-FFT of length sub_N. Identical to the
// inner kernel's local-stage twiddle layout.
inline std::pair<std::vector<float>, std::vector<float>> batch_twiddles(
    uint32_t sub_N, uint32_t log2_sub_N)
{
    const size_t total = static_cast<size_t>(log2_sub_N) * kTileElems;
    std::vector<float> r(total, 0.0f), i(total, 0.0f);
    const uint32_t num_pairs = sub_N / 2u;

    for (uint32_t s = 0; s < log2_sub_N; ++s) {
        const double M = static_cast<double>(1u << (s + 1));
        const uint32_t stride_mask = (1u << s) - 1u;
        float* tile_r = r.data() + static_cast<size_t>(s) * kTileElems;
        float* tile_i = i.data() + static_cast<size_t>(s) * kTileElems;
        for (uint32_t p = 0; p < num_pairs; ++p) {
            const uint32_t k     = p & stride_mask;
            const double   angle = -2.0 * M_PI * static_cast<double>(k) / M;
            tile_r[p] = static_cast<float>(std::cos(angle));
            tile_i[p] = static_cast<float>(std::sin(angle));
        }
    }
    return {std::move(r), std::move(i)};
}

inline std::shared_ptr<BatchFFTPlan> make_batch_plan(
    std::shared_ptr<MeshDevice> md, uint32_t sub_N, uint32_t batch)
{
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;

    auto bp = std::make_shared<BatchFFTPlan>();
    bp->md          = md;
    bp->sub_N       = sub_N;
    bp->log2_sub_N  = log2u(sub_N);
    bp->batch       = batch;

    assert(sub_N <= kTileElems   && "batch path requires sub_N <= 1024 (single tile per sub-FFT)");
    assert(is_pow2(sub_N) && sub_N >= 2);
    assert(is_pow2(batch) && batch >= 1);

    // Cap by what the device can actually accommodate as a (cols<=grid_x,
    // rows<=grid_y) rectangle. Stockham requires num_cores | batch (pow2),
    // so we use max_cores_for_grid which restricts to factorable pow2 sizes.
    const auto     dev_grid  = md->compute_with_storage_grid_size();
    const uint32_t max_cores = max_cores_for_grid(dev_grid.x, dev_grid.y);
    bp->num_cores      = (batch < max_cores) ? batch : max_cores;
    bp->batch_per_core = batch / bp->num_cores;
    assert(bp->num_cores * bp->batch_per_core == batch);
    std::tie(bp->grid_cols, bp->grid_rows) = pick_batch_grid(bp->num_cores, dev_grid.x);

    // (dev-time stdout printf removed.)

    MeshCommandQueue& cq = md->mesh_command_queue();

    // ── DRAM buffers ────────────────────────────────────────────────────
    const uint32_t io_bytes = batch * kTileSizeFp32;
    bp->in_r_buf  = make_mesh_buf(md, io_bytes, kTileSizeFp32);
    bp->in_i_buf  = make_mesh_buf(md, io_bytes, kTileSizeFp32);
    bp->out_r_buf = make_mesh_buf(md, io_bytes, kTileSizeFp32);
    bp->out_i_buf = make_mesh_buf(md, io_bytes, kTileSizeFp32);

    // ── Host scratch (reused across calls — see BatchFFTPlan comments) ─
    const size_t scratch_floats = static_cast<size_t>(batch) * kTileElems;
    bp->in_r_host.assign(scratch_floats, 0.0f);
    bp->in_i_host.assign(scratch_floats, 0.0f);
    bp->out_r_host.assign(scratch_floats, 0.0f);
    bp->out_i_host.assign(scratch_floats, 0.0f);

    auto [tw_r_data, tw_i_data] = batch_twiddles(sub_N, bp->log2_sub_N);
    const uint32_t tw_bytes = static_cast<uint32_t>(tw_r_data.size() * sizeof(float));
    bp->tw_r_buf = make_mesh_buf(md, tw_bytes, kTileSizeFp32);
    bp->tw_i_buf = make_mesh_buf(md, tw_bytes, kTileSizeFp32);
    WriteShard(cq, bp->tw_r_buf, tw_r_data, MeshCoordinate(0, 0), false);
    WriteShard(cq, bp->tw_i_buf, tw_i_data, MeshCoordinate(0, 0), false);

    // ── Program ─────────────────────────────────────────────────────────
    Program prog = CreateProgram();

    const CoreCoord first{0, 0};
    const CoreCoord last{bp->grid_cols - 1, bp->grid_rows - 1};
    const CoreRange cr(first, last);

    // CB indices match batch_fft_common.h (17 CBs, no RECV used).
    constexpr uint32_t kBatchNumCbs = 17;
    constexpr uint32_t kCbTiles[kBatchNumCbs] = {
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,   // EVEN/ODD/TW/OUT — 2-tile pipelined
        1, 1, 1, 1,                     // TMP, TW_ODD
        1, 1,                           // STATE_R, STATE_I
        1                               // SYNC
    };

    for (uint32_t id = 0; id < kBatchNumCbs; ++id) {
        CircularBufferConfig c(
            kCbTiles[id] * kTileSizeFp32,
            {{id, tt::DataFormat::Float32}});
        c.set_page_size(id, kTileSizeFp32);
        CreateCircularBuffer(prog, cr, c);
    }

    auto rk = CreateKernel(
        prog,
        "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/batch_fft_reader.cpp",
        cr,
        DataMovementConfig{
            .processor    = DataMovementProcessor::RISCV_0,
            .noc          = NOC::RISCV_0_default,
            .compile_args = {sub_N, bp->log2_sub_N}});

    auto wk = CreateKernel(
        prog,
        "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/batch_fft_writer.cpp",
        cr,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc       = NOC::RISCV_1_default});

    std::vector<UnpackToDestMode> u2d(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (uint32_t id = 0; id < kBatchNumCbs; ++id) {
        u2d[id] = UnpackToDestMode::UnpackToDestFp32;
    }

    auto ck = CreateKernel(
        prog,
        "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/compute/batch_fft_compute.cpp",
        cr,
        ComputeConfig{
            .math_fidelity       = MathFidelity::HiFi4,
            .fp32_dest_acc_en    = true,
            .unpack_to_dest_mode = u2d,
            .compile_args        = {bp->log2_sub_N}});

    for (uint32_t c = 0; c < bp->num_cores; ++c) {
        const CoreCoord  logical  = batch_logical_core(c, bp->grid_cols);
        const CoreCoord  physical = md->worker_core_from_logical_core(logical);
        const uint32_t   base     = c * bp->batch_per_core;

        SetRuntimeArgs(prog, rk, logical, {
            buf_addr(bp->in_r_buf), buf_addr(bp->in_i_buf),
            buf_addr(bp->tw_r_buf), buf_addr(bp->tw_i_buf),
            base, bp->batch_per_core,
            static_cast<uint32_t>(physical.x),
            static_cast<uint32_t>(physical.y),
        });

        SetRuntimeArgs(prog, wk, logical, {
            buf_addr(bp->out_r_buf), buf_addr(bp->out_i_buf),
            base, bp->batch_per_core,
        });

        SetRuntimeArgs(prog, ck, logical, {bp->batch_per_core});
    }

    bp->workload.add_program(
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 0)),
        std::move(prog));
    bp->initialized = true;
    return bp;
}

namespace detail {
inline std::unordered_map<uint64_t, std::shared_ptr<BatchFFTPlan>>& batch_plan_cache() {
    static std::unordered_map<uint64_t, std::shared_ptr<BatchFFTPlan>> c;
    return c;
}
inline uint64_t batch_plan_key(MeshDevice* md, uint32_t sub_N, uint32_t batch) {
    return reinterpret_cast<uint64_t>(md)
         ^ (uint64_t{sub_N} * 0x9E3779B97F4A7C15ull)
         ^ (uint64_t{batch} * 0xBF58476D1CE4E5B9ull);
}
}  // namespace detail

inline std::shared_ptr<BatchFFTPlan> get_cached_batch_plan(
    std::shared_ptr<MeshDevice> md, uint32_t sub_N, uint32_t batch)
{
    const uint64_t key = detail::batch_plan_key(md.get(), sub_N, batch);
    auto& cache = detail::batch_plan_cache();
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    auto bp = make_batch_plan(md, sub_N, batch);
    cache.emplace(key, bp);
    return bp;
}

inline void execute_batch(
    BatchFFTPlan&            plan,
    std::vector<float>&      in_r,
    std::vector<float>&      in_i,
    std::vector<float>&      out_r,
    std::vector<float>&      out_i)
{
    using namespace tt::tt_metal::distributed;
    assert(plan.initialized);
    assert(in_r.size() == static_cast<size_t>(plan.batch) * kTileElems);
    assert(in_i.size() == in_r.size());

    MeshCommandQueue& cq = plan.md->mesh_command_queue();

    WriteShard(cq, plan.in_r_buf, in_r, MeshCoordinate(0, 0), false);
    WriteShard(cq, plan.in_i_buf, in_i, MeshCoordinate(0, 0), false);

    EnqueueMeshWorkload(cq, plan.workload, false);

    ReadShard(cq, out_r, plan.out_r_buf, MeshCoordinate(0, 0), true);
    ReadShard(cq, out_i, plan.out_i_buf, MeshCoordinate(0, 0), true);
}

// Convenience: run `batch` length-`sub_N` FFTs given a flat (batch * sub_N)
// natural-order input. Handles the bit-reversal pack and the natural-order
// unpack so callers can think purely in terms of "sub-FFT i, slot j".
inline void batch_fft(
    std::shared_ptr<MeshDevice>  md,
    uint32_t                     sub_N,
    uint32_t                     batch,
    const std::vector<Complex>&  in_natural,    // size batch * sub_N
    std::vector<Complex>&        out_natural)   // resized to batch * sub_N
{
    assert(in_natural.size() == static_cast<size_t>(sub_N) * batch);

    auto plan = get_cached_batch_plan(md, sub_N, batch);
    const uint32_t log2_sub_N = plan->log2_sub_N;

    const size_t tile_floats = kTileElems;
    std::vector<float>& in_r  = plan->in_r_host;
    std::vector<float>& in_i  = plan->in_i_host;
    std::vector<float>& out_r = plan->out_r_host;
    std::vector<float>& out_i = plan->out_i_host;

    // Bit-reverse pack. Tile t holds sub-FFT t.
    for (uint32_t t = 0; t < batch; ++t) {
        const Complex* src = in_natural.data() + static_cast<size_t>(t) * sub_N;
        float* tr = in_r.data() + static_cast<size_t>(t) * tile_floats;
        float* ti = in_i.data() + static_cast<size_t>(t) * tile_floats;
        for (uint32_t k = 0; k < sub_N; ++k) {
            const uint32_t s = bit_rev(k, log2_sub_N);
            tr[k] = src[s].real();
            ti[k] = src[s].imag();
        }
    }

    execute_batch(*plan, in_r, in_i, out_r, out_i);

    out_natural.resize(static_cast<size_t>(batch) * sub_N);
    for (uint32_t t = 0; t < batch; ++t) {
        const float* tr = out_r.data() + static_cast<size_t>(t) * tile_floats;
        const float* ti = out_i.data() + static_cast<size_t>(t) * tile_floats;
        Complex* dst = out_natural.data() + static_cast<size_t>(t) * sub_N;
        for (uint32_t k = 0; k < sub_N; ++k) dst[k] = {tr[k], ti[k]};
    }
}

inline std::vector<Complex> pass1_row_ffts(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<Complex>&  x,
    const StockhamPlan&          p)
{
    assert(static_cast<uint32_t>(x.size()) == p.N);

    // Build a flat (N1 * N2) buffer where row i is the transposed slice
    // {x[0*N1 + i], x[1*N1 + i], ..., x[(N2-1)*N1 + i]}.
    std::vector<Complex> in_natural(static_cast<size_t>(p.N1) * p.N2);
    for (uint32_t i = 0; i < p.N1; ++i) {
        Complex* dst = in_natural.data() + static_cast<size_t>(i) * p.N2;
        for (uint32_t j = 0; j < p.N2; ++j) {
            dst[j] = x[static_cast<size_t>(j) * p.N1 + i];
        }
    }

    std::vector<Complex> out_natural;
    batch_fft(md, /*sub_N=*/p.N2, /*batch=*/p.N1, in_natural, out_natural);
    return out_natural;   // row-major (N1, N2) — exactly what pass 2 expects.
}

struct Pass2Plan {
    uint32_t N1 = 0, N2 = 0, N = 0;
    uint32_t num_cores      = 0;
    uint32_t tiles_per_core = 0;
    uint32_t grid_cols      = 0;
    uint32_t grid_rows      = 0;

    std::shared_ptr<MeshDevice> md;
    std::shared_ptr<MeshBuffer> in_r_buf,  in_i_buf;
    std::shared_ptr<MeshBuffer> out_r_buf, out_i_buf;
    std::shared_ptr<MeshBuffer> tw_r_buf,  tw_i_buf;
    tt::tt_metal::distributed::MeshWorkload workload;

    // Host scratch reused across calls (see the BatchFFTPlan note above for
    // rationale — same "skip the 8–16 MB memset per call" pattern).
    std::vector<float> in_r_host, in_i_host;
    std::vector<float> out_r_host, out_i_host;

    bool initialized = false;
};

// N1 tiles per side, tile i holds T[i, *] = exp(-2*pi*i*i*j/N) for j=[0,N2).
// Slots [N2, kTileElems) are zero so the SFPU mul against zero-padded input
// stays zero (we never write outside [0, N2) on the consumer side anyway).
inline std::pair<std::vector<float>, std::vector<float>> pass2_twiddle_table(
    uint32_t N1, uint32_t N2)
{
    const uint32_t N         = N1 * N2;
    const size_t   total     = static_cast<size_t>(N1) * kTileElems;
    const double   tau_over_N = -2.0 * M_PI / static_cast<double>(N);

    std::vector<float> r(total, 0.0f), i(total, 0.0f);
    for (uint32_t row = 0; row < N1; ++row) {
        float* tile_r = r.data() + static_cast<size_t>(row) * kTileElems;
        float* tile_i = i.data() + static_cast<size_t>(row) * kTileElems;
        for (uint32_t j = 0; j < N2; ++j) {
            const double angle = tau_over_N *
                                 static_cast<double>(row) *
                                 static_cast<double>(j);
            tile_r[j] = static_cast<float>(std::cos(angle));
            tile_i[j] = static_cast<float>(std::sin(angle));
        }
    }
    return {std::move(r), std::move(i)};
}

inline std::shared_ptr<Pass2Plan> make_pass2_plan(
    std::shared_ptr<MeshDevice> md, uint32_t N1, uint32_t N2)
{
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;

    auto pp = std::make_shared<Pass2Plan>();
    pp->md = md;
    pp->N1 = N1;
    pp->N2 = N2;
    pp->N  = N1 * N2;

    assert(N2 <= kTileElems && "Pass-2 device kernel requires N2 <= 1024");
    assert(is_pow2(N1) && N1 >= 2);
    assert(is_pow2(N2) && N2 >= 2);

    // Same factorisation constraint as batch_fft: num_cores | N1 (pow2).
    const auto     dev_grid  = md->compute_with_storage_grid_size();
    const uint32_t max_cores = max_cores_for_grid(dev_grid.x, dev_grid.y);
    pp->num_cores      = (N1 < max_cores) ? N1 : max_cores;
    pp->tiles_per_core = N1 / pp->num_cores;
    assert(pp->num_cores * pp->tiles_per_core == N1);
    std::tie(pp->grid_cols, pp->grid_rows) = pick_batch_grid(pp->num_cores, dev_grid.x);

    // (dev-time stdout printf removed.)

    MeshCommandQueue& cq = md->mesh_command_queue();

    const uint32_t io_bytes = N1 * kTileSizeFp32;
    pp->in_r_buf  = make_mesh_buf(md, io_bytes, kTileSizeFp32);
    pp->in_i_buf  = make_mesh_buf(md, io_bytes, kTileSizeFp32);
    pp->out_r_buf = make_mesh_buf(md, io_bytes, kTileSizeFp32);
    pp->out_i_buf = make_mesh_buf(md, io_bytes, kTileSizeFp32);

    auto [tw_r_data, tw_i_data] = pass2_twiddle_table(N1, N2);
    pp->tw_r_buf = make_mesh_buf(md, io_bytes, kTileSizeFp32);
    pp->tw_i_buf = make_mesh_buf(md, io_bytes, kTileSizeFp32);
    WriteShard(cq, pp->tw_r_buf, tw_r_data, MeshCoordinate(0, 0), false);
    WriteShard(cq, pp->tw_i_buf, tw_i_data, MeshCoordinate(0, 0), false);

    // ── Host scratch (reused across calls) ─────────────────────────────
    const size_t scratch_floats = static_cast<size_t>(N1) * kTileElems;
    pp->in_r_host.assign(scratch_floats, 0.0f);
    pp->in_i_host.assign(scratch_floats, 0.0f);
    pp->out_r_host.assign(scratch_floats, 0.0f);
    pp->out_i_host.assign(scratch_floats, 0.0f);

    Program prog = CreateProgram();

    const CoreCoord first{0, 0};
    const CoreCoord last{pp->grid_cols - 1, pp->grid_rows - 1};
    const CoreRange cr(first, last);

    // CB indices match pass2_common.h (8 CBs).
    constexpr uint32_t kPass2NumCbs = 8;
    constexpr uint32_t kCbTiles[kPass2NumCbs] = {
        2, 2, 2, 2,    // A_R, A_I, T_R, T_I — 2-tile pipeline
        2, 2,          // B_R, B_I
        1, 1           // TMP_R, TMP_I
    };

    for (uint32_t id = 0; id < kPass2NumCbs; ++id) {
        CircularBufferConfig c(
            kCbTiles[id] * kTileSizeFp32,
            {{id, tt::DataFormat::Float32}});
        c.set_page_size(id, kTileSizeFp32);
        CreateCircularBuffer(prog, cr, c);
    }

    auto rk = CreateKernel(
        prog,
        "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/pass2_reader.cpp",
        cr,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc       = NOC::RISCV_0_default});

    auto wk = CreateKernel(
        prog,
        "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/pass2_writer.cpp",
        cr,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc       = NOC::RISCV_1_default});

    std::vector<UnpackToDestMode> u2d(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (uint32_t id = 0; id < kPass2NumCbs; ++id) {
        u2d[id] = UnpackToDestMode::UnpackToDestFp32;
    }

    auto ck = CreateKernel(
        prog,
        "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/compute/pass2_compute.cpp",
        cr,
        ComputeConfig{
            .math_fidelity       = MathFidelity::HiFi4,
            .fp32_dest_acc_en    = true,
            .unpack_to_dest_mode = u2d});

    for (uint32_t c = 0; c < pp->num_cores; ++c) {
        const CoreCoord logical = batch_logical_core(c, pp->grid_cols);
        const uint32_t  base    = c * pp->tiles_per_core;

        SetRuntimeArgs(prog, rk, logical, {
            buf_addr(pp->in_r_buf), buf_addr(pp->in_i_buf),
            buf_addr(pp->tw_r_buf), buf_addr(pp->tw_i_buf),
            base, pp->tiles_per_core,
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

namespace detail {
inline std::unordered_map<uint64_t, std::shared_ptr<Pass2Plan>>& pass2_plan_cache() {
    static std::unordered_map<uint64_t, std::shared_ptr<Pass2Plan>> c;
    return c;
}
inline uint64_t pass2_plan_key(MeshDevice* md, uint32_t N1, uint32_t N2) {
    return reinterpret_cast<uint64_t>(md)
         ^ (uint64_t{N1} * 0xD1B54A32D192ED03ull)
         ^ (uint64_t{N2} * 0xAEF17502108EF2D9ull);
}
}  // namespace detail

inline std::shared_ptr<Pass2Plan> get_cached_pass2_plan(
    std::shared_ptr<MeshDevice> md, uint32_t N1, uint32_t N2)
{
    const uint64_t key = detail::pass2_plan_key(md.get(), N1, N2);
    auto& cache = detail::pass2_plan_cache();
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    auto pp = make_pass2_plan(md, N1, N2);
    cache.emplace(key, pp);
    return pp;
}

inline std::vector<Complex> pass2_twiddle_transpose(
    std::shared_ptr<MeshDevice> md,
    const std::vector<Complex>& A,
    const StockhamPlan&         p)
{
    using namespace tt::tt_metal::distributed;
    auto plan = get_cached_pass2_plan(md, p.N1, p.N2);

    // Reuse the plan's host scratch (pre-zeroed once at plan build time —
    // the kernel only reads [0, N2) per row, so the padded region stays
    // zero for the life of the plan and no per-call memset is needed).
    std::vector<float>& in_r  = plan->in_r_host;
    std::vector<float>& in_i  = plan->in_i_host;
    std::vector<float>& out_r = plan->out_r_host;
    std::vector<float>& out_i = plan->out_i_host;

    for (uint32_t i = 0; i < p.N1; ++i) {
        const Complex* src = A.data() + static_cast<size_t>(i) * p.N2;
        float* tr = in_r.data() + static_cast<size_t>(i) * kTileElems;
        float* ti = in_i.data() + static_cast<size_t>(i) * kTileElems;
        for (uint32_t j = 0; j < p.N2; ++j) {
            tr[j] = src[j].real();
            ti[j] = src[j].imag();
        }
    }

    MeshCommandQueue& cq = md->mesh_command_queue();
    WriteShard(cq, plan->in_r_buf, in_r, MeshCoordinate(0, 0), false);
    WriteShard(cq, plan->in_i_buf, in_i, MeshCoordinate(0, 0), false);
    EnqueueMeshWorkload(cq, plan->workload, false);

    ReadShard(cq, out_r, plan->out_r_buf, MeshCoordinate(0, 0), true);
    ReadShard(cq, out_i, plan->out_i_buf, MeshCoordinate(0, 0), true);

    // Transpose B (N1, N2) → C (N2, N1) on host. Pure memory shuffling,
    // ~10 ms at N=1M; trivial vs. the cos/sin we just eliminated.
    std::vector<Complex> C(p.N);
    for (uint32_t i = 0; i < p.N1; ++i) {
        const float* tr = out_r.data() + static_cast<size_t>(i) * kTileElems;
        const float* ti = out_i.data() + static_cast<size_t>(i) * kTileElems;
        for (uint32_t j = 0; j < p.N2; ++j) {
            C[static_cast<size_t>(j) * p.N1 + i] = {tr[j], ti[j]};
        }
    }
    return C;
}

inline std::vector<Complex> pass3_row_ffts(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<Complex>&  C,
    const StockhamPlan&          p)
{
    assert(static_cast<uint32_t>(C.size()) == p.N);
    // C is already (N2, N1) row-major: rows are sub-FFT inputs in natural
    // order. Hand the buffer straight to the batch dispatcher.
    std::vector<Complex> D;
    batch_fft(md, /*sub_N=*/p.N1, /*batch=*/p.N2, C, D);
    return D;
}

// ── Final reorder: D is (N2, N1) row-major; natural 1D output:
//     X[k] = D[k % N2, k / N2] = D_flat[(k % N2) * N1 + (k / N2)]
inline std::vector<Complex> final_reorder(
    const std::vector<Complex>& D,
    const StockhamPlan&         p)
{
    std::vector<Complex> X(p.N);
    for (uint32_t k = 0; k < p.N; ++k) {
        const uint32_t j  = k % p.N2;
        const uint32_t ip = k / p.N2;
        X[k] = D[static_cast<size_t>(j) * p.N1 + ip];
    }
    return X;
}

inline std::vector<Complex> fft(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<Complex>&  signal)
{
    const uint32_t N = static_cast<uint32_t>(signal.size());
    assert(N >= 2 && "FFT requires N >= 2");
    assert(is_pow2(N) && "FFT requires N to be a power of two");

    const StockhamPlan p = plan(N);

    if (!p.stockham) {
        return fft_example::fft(md, signal);
    }

    // (dev-time stdout printf removed.)

    const auto A = pass1_row_ffts        (md, signal, p);
    const auto C = pass2_twiddle_transpose(md, A,      p);
    const auto D = pass3_row_ffts        (md, C,      p);
    return final_reorder(D, p);
}

inline std::vector<Complex> fft(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<float>&    signal)
{
    std::vector<Complex> cx(signal.size());
    for (size_t i = 0; i < signal.size(); ++i) cx[i] = {signal[i], 0.0f};
    return fft(md, cx);
}

}  // namespace fft_stockham
