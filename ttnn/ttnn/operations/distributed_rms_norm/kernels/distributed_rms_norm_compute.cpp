// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// distributed_rms_norm compute (op_design.md -> Block schedule). One block per core (the whole assignment resident),
// cut into NUM_WAVES row waves of Q_W tile-rows.
//
// Phase A (needs no remote data), per wave w:
//   square_collapse_wave   eltwise_chain x*x DEST-accumulated per tile-row -> cb_sumsq_partial, then
//                          reduce<SUM, REDUCE_ROW> -> cb_partial_col (D3: the slice partial for the owner) or
//                          cb_chip_partial (D1: the chip partial itself). Sharded W-pad: the base's partial-scaler
//                          mask on the core holding tile Wt-1.
//   chip_sum_wave (owner)  reduce<SUM, REDUCE_COL, AccumulateViaAdd, Skip> over the G_W gathered partials ->
//                          cb_chip_partial — lagged one wave behind the square so the gather of wave w lands while
//                          wave w+1 is squared.
// scale_by_gamma_block     x * γ (Row broadcast) for the whole block -> cb_x_gamma (the stall shadow of the exchange)
// Phase B, per wave w:
//   cross_device_finalize  (owner) fold the G devices' records of each owned row in group-position order — SFPU fp32
//                          adds, 2 DEST slots — then rsqrt(ss/W + eps) on column 0 -> cb_rms_root
//   scale_by_inv_rms_wave  (x*γ | x) * inv_rms (Col broadcast) -> cb_output
//
// Raw-LLK uses (helper mismatch, op_design.md -> "Helpers considered and rejected"):
//   cross-device fold: reduce<AccumulateViaAdd> adds through SrcA/SrcB (tf32) — the exchanged partials must be summed
//     in fp32; a chain cannot express a G-way fold of G pages of one CB into one accumulator. copy_tile (with
//     UnpackToDestFp32 on cb_stats) + add_binary_tile (SFPU) keeps fp32 in DEST with 2 live slots for any G.
//   finalize: the base op's finalize_col0_body (one fused SFPU pass over column 0, base Refinement 6 / lamp L7).

#include <stdint.h>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

// Measurement-only stage zones (host DRMS_PERF_ZONES=1 -> define DRMS_PERF_ZONES; off in production).
#if defined(DRMS_PERF_ZONES) && DRMS_PERF_ZONES
#include "tools/profiler/kernel_profiler.hpp"
#define DRMS_ZONE(name) DeviceZoneScopedN(name)
#else
#define DRMS_ZONE(name)
#endif
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_converter.h"
#endif

namespace ckl = compute_kernel_lib;

namespace {

#ifdef TRISC_MATH
// Base op finalize_col0_body: inv_rms = rsqrt(ss * (1/W) + eps) over the even column-0 vectors of faces 0/2.
sfpi_inline void finalize_col0_body(const uint32_t inv_w_bits, const uint32_t eps_bits) {
#pragma GCC unroll 0
    for (int rg = 0; rg < 4; rg++) {
        const sfpi::vFloat mean_sq = sfpi::dst_reg[0] * ckernel::sfpu::Converter::as_float(inv_w_bits) +
                                     ckernel::sfpu::Converter::as_float(eps_bits);
        sfpi::vFloat t = ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false>(mean_sq);
        if constexpr (!DST_ACCUM_MODE) {
            t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg += 2;
    }
}
#endif

constexpr uint32_t cb_x_tiles = 1;
constexpr uint32_t cb_reduce_scaler = 2;
constexpr uint32_t cb_sumsq_partial = 3;
constexpr uint32_t cb_inv_rms = 4;
constexpr uint32_t cb_gamma_tiles = 6;
constexpr uint32_t cb_x_gamma = 7;
constexpr uint32_t cb_partial_col = 9;
constexpr uint32_t cb_gather = 10;
constexpr uint32_t cb_rms_root = 11;
constexpr uint32_t cb_sumsq_last = 12;
constexpr uint32_t cb_sumsq_acc = 13;
constexpr uint32_t cb_chip_partial = 14;
constexpr uint32_t cb_stats = 15;
constexpr uint32_t cb_output = 16;

}  // namespace

void kernel_main() {
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(0) != 0;
    constexpr bool SHARDED = get_compile_time_arg_val(1) != 0;
    constexpr bool COMBINE = get_compile_time_arg_val(2) != 0;
    constexpr bool MASKED_CFG = get_compile_time_arg_val(3) != 0;
    constexpr uint32_t WS = get_compile_time_arg_val(4);
    constexpr uint32_t C = get_compile_time_arg_val(5);
    constexpr uint32_t NCH = get_compile_time_arg_val(6);
    constexpr uint32_t NW = get_compile_time_arg_val(7);
    constexpr uint32_t Q_W = get_compile_time_arg_val(8);
    constexpr uint32_t OWNERS = get_compile_time_arg_val(9);
    constexpr uint32_t Q = get_compile_time_arg_val(10);
    constexpr uint32_t G = get_compile_time_arg_val(11);
    constexpr uint32_t G_W = get_compile_time_arg_val(12);
    constexpr uint32_t SCALE_BLOCK = get_compile_time_arg_val(13);
    // output placement: shard-backed cb_output (packed in place, strided) or streamed to the writer per tile-row
    constexpr bool OUT_SHARDED = get_compile_time_arg_val(14) != 0;
    // γ pass per wave instead of per block (L1 fallback: cb_x_gamma holds Q_W rows instead of the block)
    constexpr bool GAMMA_PER_WAVE = get_compile_time_arg_val(15) != 0;
    constexpr uint32_t WP = NCH * C;

    const uint32_t rows = get_arg_val<uint32_t>(0);
    const uint32_t V = get_arg_val<uint32_t>(1);
    const uint32_t my_j = get_arg_val<uint32_t>(2);
    const bool masked_core = get_arg_val<uint32_t>(3) != 0;
    const uint32_t inv_w_bits = get_arg_val<uint32_t>(4);
    const uint32_t eps_bits = get_arg_val<uint32_t>(5);

    compute_kernel_hw_startup(cb_x_tiles, cb_x_tiles, cb_sumsq_partial);

    constexpr uint32_t cb_collapsed = COMBINE ? cb_partial_col : cb_chip_partial;

    auto n_wave = [&](uint32_t w) -> uint32_t {
        const uint32_t r0 = w * Q_W;
        return r0 >= rows ? 0 : ((rows - r0) < Q_W ? rows - r0 : Q_W);
    };
    // my owner slot k in wave w (owner k of wave w = member (w*OWNERS + k) mod G_W), or OWNERS if none
    auto my_k = [&](uint32_t w) -> uint32_t {
        for (uint32_t k = 0; k < OWNERS; ++k) {
            if ((w * OWNERS + k) % G_W == my_j) {
                return k;
            }
        }
        return OWNERS;
    };
    auto n_own = [&](uint32_t w, uint32_t k) -> uint32_t {
        const uint32_t n = n_wave(w);
        const uint32_t first = k * Q;
        return first >= n ? 0 : ((n - first) < Q ? n - first : Q);
    };

    using SumsqPack = ckl::PackTile<ckl::output(
        cb_sumsq_partial,
        ckl::ReservePolicy::PerOuter,
        ckl::PushPolicy::PerOuter,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Direct,
        ckl::DestAccumulation::PerRow)>;
    constexpr auto x_strided_input = ckl::input(
        cb_x_tiles,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Block,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Strided);
    using SquareAccumulateStrided = ckl::
        BinaryFpu<ckl::BinaryFpuOp::Mul, x_strided_input, x_strided_input, ckl::Dst::D0, ckl::DestAccumulation::PerRow>;
    constexpr auto x_gamma_strided_input = ckl::input(
        cb_x_gamma,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Block,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Strided);
    constexpr auto inv_rms_strided_input = ckl::input(
        cb_inv_rms,
        ckl::BroadcastDim::Col,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Col,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Strided);
    constexpr auto row_output = [](uint32_t cb) {
        return ckl::output(cb, ckl::ReservePolicy::PerOuter, ckl::PushPolicy::PerOuter);
    };
    constexpr auto shard_output = ckl::output(
        cb_output,
        ckl::ReservePolicy::None,
        ckl::PushPolicy::None,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Strided);

    auto collapse = [&](uint32_t n, auto partial_scaler, auto accumulate, auto cb_in_tag, auto cb_out_tag) {
        constexpr uint32_t cb_in = decltype(cb_in_tag)::value;
        constexpr uint32_t cb_out = decltype(cb_out_tag)::value;
        ckl::reduce<
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            cb_in,
            cb_reduce_scaler,
            cb_out,
            ckl::ReduceInputPolicy::BulkWaitBulkPop>(
            ckl::ReduceInputBlockShape::of(n, 1),
            ckl::ReduceInputMemoryLayout::contiguous(),
            accumulate,
            ckl::NoOp{},
            partial_scaler);
    };
    using CbSumsqPartial = std::integral_constant<uint32_t, cb_sumsq_partial>;
    using CbSumsqLast = std::integral_constant<uint32_t, cb_sumsq_last>;
    using CbSumsqAcc = std::integral_constant<uint32_t, cb_sumsq_acc>;
    using CbCollapsed = std::integral_constant<uint32_t, cb_collapsed>;

    const uint32_t full_tiles = masked_core ? V - 1 : V;

    // ---- square_collapse_wave ----
    auto square_collapse = [&](uint32_t w) {
        const uint32_t n = n_wave(w);
        if (n == 0) {
            return;
        }
        const uint32_t r0 = w * Q_W;
        const uint32_t xbase = r0 * WS;
        if constexpr (!SHARDED) {
            DRMS_ZONE("cmp_wait_x");
            cb_wait_front(cb_x_tiles, (r0 + n) * WS);
        }
        DRMS_ZONE("cmp_square");
        if (full_tiles > 0) {
            ckl::eltwise_chain(
                ckl::IterationShape::grid(n, full_tiles),
                SquareAccumulateStrided{ckl::StridedTileRange{xbase, WS}, ckl::StridedTileRange{xbase, WS}},
                SumsqPack{});
        }
        if constexpr (MASKED_CFG) {
            if (masked_core) {
                const ckl::StridedTileRange last_tile{xbase + V - 1, WS};
                ckl::eltwise_chain(
                    ckl::IterationShape::grid(n, 1),
                    SquareAccumulateStrided{last_tile, last_tile},
                    ckl::PackTile<ckl::output(
                        cb_sumsq_last,
                        ckl::ReservePolicy::PerOuter,
                        ckl::PushPolicy::PerOuter,
                        ckl::DataFormatReconfig::Enabled,
                        ckl::TileAddressing::Direct,
                        ckl::DestAccumulation::PerRow)>{});
                if (full_tiles > 0) {
                    collapse(
                        n, ckl::ReducePartialScaler::none(), ckl::NoAccumulation{}, CbSumsqPartial{}, CbSumsqAcc{});
                    collapse(
                        n,
                        ckl::ReducePartialScaler::with_partial(),
                        ckl::Accumulate::at(cb_sumsq_acc, 1),
                        CbSumsqLast{},
                        CbCollapsed{});
                } else {
                    collapse(
                        n,
                        ckl::ReducePartialScaler::with_partial(),
                        ckl::NoAccumulation{},
                        CbSumsqLast{},
                        CbCollapsed{});
                }
                return;
            }
        }
        collapse(n, ckl::ReducePartialScaler::none(), ckl::NoAccumulation{}, CbSumsqPartial{}, CbCollapsed{});
    };

    // ---- chip_sum_wave (owner): Σ over the G_W slice partials of the owned rows ----
    auto chip_sum = [&](uint32_t w) {
        if constexpr (COMBINE) {
            const uint32_t k = my_k(w);
            if (k == OWNERS) {
                return;
            }
            const uint32_t no = n_own(w, k);
            if (no == 0) {
                return;
            }
            DRMS_ZONE("cmp_chipsum");
            ckl::reduce<
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_COL,
                cb_gather,
                cb_reduce_scaler,
                cb_chip_partial,
                ckl::ReduceInputPolicy::BulkWaitBulkPop,
                ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                ReduceFp32Mode::Fast,
                ckl::ReduceAlgorithm::AccumulateViaAdd,
                ckl::ReduceWithinTile::Skip>(
                ckl::ReduceInputBlockShape::of(G_W, no),
                ckl::ReduceInputMemoryLayout::with_row_stride(Q),
                ckl::NoAccumulation{},
                ckl::NoOp{});
        }
    };

    // ---- cross_device_finalize (owner): position-order fp32 fold of the G records + column-0 finalize ----
    auto fold_finalize = [&](uint32_t no) {
        {
            DRMS_ZONE("cmp_wait_stats");
            cb_wait_front(cb_stats, G * Q);
        }
        DRMS_ZONE("cmp_fold");
        reconfig_data_format_srca(cb_stats);
        pack_reconfig_data_format(cb_rms_root);
        cb_reserve_back(cb_rms_root, no);
        copy_init(cb_stats);
        add_binary_tile_init();
        rsqrt_tile_init();  // the sqrt body's programmable constants
        for (uint32_t r = 0; r < no; ++r) {
            tile_regs_acquire();
            copy_tile(cb_stats, r, 0);
            for (uint32_t pos = 1; pos < G; ++pos) {
                copy_tile(cb_stats, pos * Q + r, 1);
                add_binary_tile(0, 1, 0);
            }
            MATH((_llk_math_eltwise_unary_sfpu_params_(
                finalize_col0_body, 0, ckernel::VectorMode::C, inv_w_bits, eps_bits)));
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_rms_root);
            tile_regs_release();
        }
        cb_push_back(cb_rms_root, no);
        cb_pop_front(cb_stats, G * Q);
    };

    if constexpr (SHARDED) {
        cb_wait_front(cb_x_tiles, rows * WS);
    }

    // ================= Phase A =================
    for (uint32_t w = 0; w <= NW; ++w) {
        if (w < NW) {
            square_collapse(w);
        }
        if (w >= 1) {
            chip_sum(w - 1);
        }
    }

    // ================= scale_by_gamma =================
    // x * γ for tile-rows [row0, row0 + n) -> cb_x_gamma (WS tiles pushed per row).
    auto gamma_pass = [&](uint32_t row0, uint32_t n) {
        if constexpr (HAS_GAMMA) {
            {
                DRMS_ZONE("cmp_wait_gamma");
                cb_wait_front(cb_gamma_tiles, WP);
            }
            DRMS_ZONE("cmp_gamma");
            ckl::eltwise_chain(
                ckl::IterationShape::grid(n, WS).block_size(SCALE_BLOCK),
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Mul,
                    x_strided_input,
                    ckl::input(
                        cb_gamma_tiles,
                        ckl::BroadcastDim::Row,
                        ckl::WaitPolicy::None,
                        ckl::PopPolicy::None,
                        ckl::InputTileMapping::Row,
                        ckl::DataFormatReconfig::Enabled,
                        ckl::TileAddressing::Offset)>{ckl::StridedTileRange{row0 * WS, WS}, 0u},
                ckl::PackTile<row_output(cb_x_gamma)>{});
        }
    };
    // Whole block (default): the γ pass of every wave is the stall shadow of the exchange. Per wave (L1 fallback,
    // GAMMA_PER_WAVE: cb_x_gamma holds one wave): it runs right before the wave's inv scaling.
    if constexpr (HAS_GAMMA && !GAMMA_PER_WAVE) {
        gamma_pass(0, rows);
        cb_wait_front(cb_x_gamma, rows * WS);
    }

    // ================= Phase B =================
    if constexpr (OUT_SHARDED) {
        cb_reserve_back(cb_output, rows * WS);
    }
    // cross_device_finalize of every owned wave first (in wave order): an owner's fold of wave w+1 never waits
    // behind its scaling of wave w, which waits for another owner's all-gather.
    for (uint32_t w = 0; w < NW; ++w) {
        const uint32_t k = my_k(w);
        if (k < OWNERS) {
            const uint32_t no = n_own(w, k);
            if (no > 0) {
                fold_finalize(no);
            }
        }
    }
    for (uint32_t w = 0; w < NW; ++w) {
        const uint32_t n = n_wave(w);
        const uint32_t r0 = w * Q_W;
        if constexpr (HAS_GAMMA && GAMMA_PER_WAVE) {
            if (n > 0) {
                gamma_pass(r0, n);
                cb_wait_front(cb_x_gamma, n * WS);
            }
        }
        {
            DRMS_ZONE("cmp_wait_inv");
            cb_wait_front(cb_inv_rms, (w + 1) * Q_W);
        }
        DRMS_ZONE("cmp_scale");
        if (n == 0) {
            continue;
        }
        // x*γ rows: block mode addresses the wave inside the resident block; per-wave mode the wave is the front
        const uint32_t xg0 = GAMMA_PER_WAVE ? 0 : r0 * WS;
        const ckl::StridedTileRange inv_range{r0, 1};
        for (uint32_t c = 0; c < NCH; ++c) {
            const uint32_t cw = (c + 1 == NCH) ? WS - c * C : C;
            const ckl::StridedTileRange a_range{r0 * WS + c * C, WS};
            [[maybe_unused]] const ckl::StridedTileRange xg_range{xg0 + c * C, WS};
            if constexpr (OUT_SHARDED) {
                if constexpr (HAS_GAMMA) {
                    ckl::eltwise_chain(
                        ckl::IterationShape::grid(n, cw).block_size(SCALE_BLOCK),
                        ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_gamma_strided_input, inv_rms_strided_input>{
                            xg_range, inv_range},
                        ckl::PackTile<shard_output>{a_range});
                } else {
                    ckl::eltwise_chain(
                        ckl::IterationShape::grid(n, cw).block_size(SCALE_BLOCK),
                        ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_strided_input, inv_rms_strided_input>{
                            a_range, inv_range},
                        ckl::PackTile<shard_output>{a_range});
                }
            } else {
                if constexpr (HAS_GAMMA) {
                    ckl::eltwise_chain(
                        ckl::IterationShape::grid(n, cw).block_size(SCALE_BLOCK),
                        ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_gamma_strided_input, inv_rms_strided_input>{
                            xg_range, inv_range},
                        ckl::PackTile<row_output(cb_output)>{});
                } else {
                    ckl::eltwise_chain(
                        ckl::IterationShape::grid(n, cw).block_size(SCALE_BLOCK),
                        ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_strided_input, inv_rms_strided_input>{
                            a_range, inv_range},
                        ckl::PackTile<row_output(cb_output)>{});
                }
            }
        }
        if constexpr (HAS_GAMMA && GAMMA_PER_WAVE) {
            cb_pop_front(cb_x_gamma, n * WS);
        }
    }
    if constexpr (OUT_SHARDED) {
        cb_push_back(cb_output, rows * WS);
    }

    // ================= release_block =================
    cb_pop_front(cb_x_tiles, rows * WS);
    cb_pop_front(cb_inv_rms, NW * Q_W);
    if constexpr (HAS_GAMMA) {
        if constexpr (!GAMMA_PER_WAVE) {
            cb_pop_front(cb_x_gamma, rows * WS);
        }
        cb_pop_front(cb_gamma_tiles, WP);
    }
    cb_pop_front(cb_reduce_scaler, MASKED_CFG ? 2 : 1);
}
