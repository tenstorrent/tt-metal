// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// stats_fusion bake-off: the variance stage (c = z - mean is already resident in cb_centered) ->
// inv_std, in four variants selected by STATS_VARIANT. Shared by the isolated bench and the whole-op
// variant kernels so both measure the exact same code.
//
//   0 baseline    : row_sumsq chain (mul DEST-accumulate -> fp32 sumsq CB) + reduce() helper
//                   (copy_tile seed, sfpu_reduce ROW, post-op: *1/W, +eps, rsqrt = 3 SFPU calls).
//   1 tail_fold   : one DEST window per row: mul DEST-accumulate, then sfpu_reduce ROW and the same
//                   3-call finalize on that DEST tile, pack inv_std. No sumsq CB round trip.
//   2 tail_fused  : tail_fold with the finalize as ONE SFPU pass: rsqrt(S * 1/W + eps) via
//                   calculate_add_rsqrt (one SFPMAD for the scale+eps, one init).
//   3 fused_only  : baseline structure, post-op replaced by the one-pass add_rsqrt.
//   4 tail_col0   : tail_fold with a column-0-only fused finalize: rsqrt(S * 1/W + eps) evaluated only
//                   on the SFPU loads that hold tile column 0 (even-column loads, 4 of the 8 per face),
//                   half the rsqrt work of VectorMode::C. normalize's col broadcast reads only column 0.
//   5 col0_only   : baseline structure, post-op replaced by the column-0-only fused finalize.
//
// Raw LLK note (variants 1, 2): bypasses eltwise_chain + reduce(). The chain's PerRow DEST
// accumulation packs the accumulated tile at row end and has no post-accumulation hook, and reduce()
// only reads its input from a CB. Folding the collapse into the accumulating window therefore needs
// the window written out by hand: capability gap, not ergonomics.

#pragma once

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/experimental/add_rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#ifndef STATS_VARIANT
#define STATS_VARIANT 0
#endif

// Diagnostic sub-zones (bench only). Zone ends are not synchronized with the SFPU/FPU pipeline, so
// they attribute issue time; the enclosing zone total is the trustworthy number.
#ifdef STATS_DIAG
#define STATS_DIAG_ZONE(name) DeviceZoneScopedN(name)
#else
#define STATS_DIAG_ZONE(name)
#endif

#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {
// One face of rsqrt(x * scale + addend), only on the SFPU loads holding column 0. An SFPU load covers
// 4 rows x 8 columns (dest address bit 1 picks the even / odd columns), dst_reg[k] is address 2k, so
// column 0 sits in dst_reg[0], [2], [4], [6]. VectorMode::C runs this on faces 0 and 2 (rows 0-31).
template <bool APPROXIMATION_MODE, bool fp32_dest_acc_en, uint32_t scale_bits>
inline void calculate_add_rsqrt_col0(uint32_t addend_bits) {
    const sfpi::vFloat scale = Converter::as_float(scale_bits);
    const sfpi::vFloat addend = Converter::as_float(addend_bits);
#pragma GCC unroll 4
    for (int group = 0; group < 4; ++group) {
        sfpi::vFloat x = sfpi::dst_reg[2 * group];
        sfpi::vFloat y = _calculate_sqrt_body_<APPROXIMATION_MODE, true, false>(x * scale + addend);
        if constexpr (!fp32_dest_acc_en) {
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[2 * group] = y;
    }
}
}  // namespace ckernel::sfpu
#endif

namespace ln_stats {

// Baseline finalize: S * 1/W, + eps, rsqrt as three SFPU calls (column 0 only: VectorMode::C).
template <uint32_t inv_w_bits, uint32_t epsilon_bits>
ALWI void finalize_three_pass(uint32_t dst_idx) {
    binop_with_scalar_tile_init();
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, MUL_UNARY, 8, DST_ACCUM_MODE),
        dst_idx,
        VectorMode::C,
        inv_w_bits));
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, ADD_UNARY, 8, DST_ACCUM_MODE),
        dst_idx,
        VectorMode::C,
        epsilon_bits));
    rsqrt_tile_init();
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_rsqrt, (APPROX, 8, DST_ACCUM_MODE, false), dst_idx, VectorMode::C));
}

// One SFPU pass: rsqrt(S * 1/W + eps). Same 23-bit rsqrt body as rsqrt_tile (APPROX follows
// math_approx_mode); the scale + eps is a single fused multiply-add.
template <uint32_t inv_w_bits, uint32_t epsilon_bits>
ALWI void finalize_fused(uint32_t dst_idx) {
#ifdef ARCH_BLACKHOLE
    add_rsqrt_tile_init();
    add_rsqrt_tile<false, VectorMode::C, 8, false, inv_w_bits>(dst_idx, epsilon_bits);
#else
    finalize_three_pass<inv_w_bits, epsilon_bits>(dst_idx);
#endif
}

template <uint32_t inv_w_bits, uint32_t epsilon_bits>
ALWI void finalize_col0(uint32_t dst_idx) {
    rsqrt_tile_init();  // same sqrt_init constants as add_rsqrt_tile_init, and available on WH too
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_add_rsqrt_col0,
        (APPROX, DST_ACCUM_MODE, inv_w_bits),
        dst_idx,
        VectorMode::C,
        epsilon_bits));
}

template <uint32_t inv_w_bits, uint32_t epsilon_bits>
ALWI void finalize(uint32_t dst_idx) {
    if constexpr (STATS_VARIANT == 4 || STATS_VARIANT == 5) {
        finalize_col0<inv_w_bits, epsilon_bits>(dst_idx);
    } else if constexpr (STATS_VARIANT == 2 || STATS_VARIANT == 3) {
        finalize_fused<inv_w_bits, epsilon_bits>(dst_idx);
    } else {
        finalize_three_pass<inv_w_bits, epsilon_bits>(dst_idx);
    }
}

// c (rows x Wt, at the front of cb_centered, left resident) -> inv_std (rows tiles, column 0 valid).
template <
    uint32_t cb_centered,
    uint32_t cb_row_sumsq,
    uint32_t cb_scaler,
    uint32_t cb_inv_std,
    uint32_t tensor_w_tiles,
    uint32_t inv_w_bits,
    uint32_t epsilon_bits>
ALWI void inv_std_block(uint32_t rows, DataflowBuffer& centered, DataflowBuffer& row_sumsq, DataflowBuffer& inv_std) {
    using namespace compute_kernel_lib;
    if constexpr (STATS_VARIANT == 0 || STATS_VARIANT == 3 || STATS_VARIANT == 5) {
        constexpr InputSpec centered_resident =
            input(cb_centered, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block);
        using SquareAccumulate =
            BinaryFpu<BinaryFpuOp::Mul, centered_resident, centered_resident, Dst::D0, DestAccumulation::PerRow>;
        using SumsqPack = PackTile<output(
            cb_row_sumsq,
            ReservePolicy::PerOuter,
            PushPolicy::PerOuter,
            DataFormatReconfig::Enabled,
            TileAddressing::Direct,
            DestAccumulation::PerRow)>;
        {
            MaybeDeviceZoneScope("compute_row_sumsq");
            centered.wait_front(rows * tensor_w_tiles);
            eltwise_chain(
                IterationShape::grid(rows, tensor_w_tiles), SquareAccumulate{centered, centered}, SumsqPack{row_sumsq});
        }
        {
            MaybeDeviceZoneScope("compute_inv_std");
            reduce<
                PoolType::SUM,
                ReduceDim::REDUCE_ROW,
                cb_row_sumsq,
                cb_scaler,
                cb_inv_std,
                ReduceInputPolicy::BulkWaitBulkPop,
                ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                ReduceFp32Mode::Fast,
                ReduceAlgorithm::AccumulateViaAdd,
                ReduceWithinTile::Collapse>(
                ReduceInputBlockShape::of(rows, 1),
                ReduceInputMemoryLayout::contiguous(),
                NoAccumulation{},
                [](uint32_t dst_idx) { finalize<inv_w_bits, epsilon_bits>(dst_idx); });
        }
    } else {
        MaybeDeviceZoneScope("compute_sumsq_inv_std");
        constexpr DataFormat dst_fmt = DST_ACCUM_MODE ? DataFormat::Float32 : DataFormat::Float16_b;
        centered.wait_front(rows * tensor_w_tiles);
        reconfig_data_format(cb_centered, cb_centered);
        pack_reconfig_data_format(cb_inv_std);
        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t base = row * tensor_w_tiles;
            tile_regs_acquire();
            {
                STATS_DIAG_ZONE("diag_mul");
                // acc_to_dest: a freshly acquired DEST reads 0, so the first mul seeds the sum.
                mul_init(cb_centered, cb_centered, true);
                for (uint32_t col = 0; col < tensor_w_tiles; ++col) {
                    mul_tiles(cb_centered, cb_centered, base + col, base + col, 0);
                }
            }
            {
                STATS_DIAG_ZONE("diag_collapse");
                sfpu_reduce_init<PoolType::SUM, dst_fmt>();
                sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_ROW>(0, 1, 1);
            }
            {
                STATS_DIAG_ZONE("diag_finalize");
                finalize<inv_w_bits, epsilon_bits>(0);
            }
            tile_regs_commit();
            inv_std.reserve_back(1);
            tile_regs_wait();
            pack_tile(0, cb_inv_std);
            tile_regs_release();
            inv_std.push_back(1);
        }
    }
}

}  // namespace ln_stats
