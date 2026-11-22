// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.bringup.rms_norm, the split program (CHANGELOG 6): the options `stats_out`, `partial_stats`,
// `input_window` and `dtype`.  One work unit = (row tile r, column block b) of the input's column window.
//
//   MODE_LOCAL      pass A: x^2 over the unit's window tiles, packer L1-accumulated into one fp32 tile
//                   (every column a partial), reduced along the row (SUM, scaler 1.0) -> s; then pass B.
//   MODE_STATS_OUT  pass A only: the L1-accumulated fp32 tile IS the output (a partial sum of squares in every
//                   column of the tile, over the unit's column block).
//   MODE_STATS_IN   s = the row sum of ALL columns of the G gathered partial tiles (every column of every tile is
//                   a partial sum of squares), then pass B.
//   finalize (in DEST, fp32 SFPU):  rs = rsqrt(s * (1 / stats_width) + eps)
//   pass B:  y = (x * rs[col bcast], FPU) [* gamma[row bcast], SFPU in DEST]  -> the output CB (its dtype).
//   HAS_TWIN (bfp8_twin, P.21): also y rounded to bf16 in DEST (ties away from zero, the packer's bf16 rounding,
//            DS41-103) into the free DEST half, packed to the bfloat8_b twin CB (bfp8_pack_precise: one rounding of
//            bf16-exact values = ttnn.typecast(bf16 output, bfloat8_b)).

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"

#ifdef TRISC_MATH
#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
namespace rms_twin {
// DEST tile t -> tile t + OFF: the fp32 value rounded to bf16, ties away from zero (sign-magnitude: adding half an
// ulp of the kept 16 bits rounds the magnitude of either sign), as the packer rounds fp32 -> bf16.
template <uint32_t OFF>
inline void round_bf16_away() {
    for (int d = 0; d < 8; ++d) {
        sfpi::vUInt u = sfpi::as<sfpi::vUInt>(sfpi::vFloat(sfpi::dst_reg[0]));
        u = u + sfpi::vUInt(0x8000);
        u = u & sfpi::vUInt(0xFFFF0000u);
        sfpi::dst_reg[OFF * 32] = sfpi::as<sfpi::vFloat>(u);
        sfpi::dst_reg++;
    }
}
}  // namespace rms_twin
#endif

constexpr uint32_t MODE_LOCAL = 0;
constexpr uint32_t MODE_STATS_OUT = 1;
constexpr uint32_t MODE_STATS_IN = 2;

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(1);
    constexpr uint32_t cb_acc = get_compile_time_arg_val(2);
    constexpr uint32_t cb_stats = get_compile_time_arg_val(3);
    constexpr uint32_t cb_rs = get_compile_time_arg_val(4);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(5);
    constexpr uint32_t cb_tmp = get_compile_time_arg_val(6);
    constexpr uint32_t cb_out = get_compile_time_arg_val(7);
    constexpr uint32_t WB = get_compile_time_arg_val(8);   // tiles per unit (column block)
    constexpr uint32_t BLK = get_compile_time_arg_val(9);  // DEST block, divides WB
    constexpr uint32_t G = get_compile_time_arg_val(10);   // gathered partial tiles per row (MODE_STATS_IN)
    constexpr uint32_t MODE = get_compile_time_arg_val(11);
    constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(12);
    constexpr uint32_t INV_W_BITS = get_compile_time_arg_val(13);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(14);
    constexpr uint32_t cb_twin = get_compile_time_arg_val(15);
    constexpr uint32_t HAS_TWIN = get_compile_time_arg_val(16);

    const uint32_t num_units = get_arg_val<uint32_t>(0);

    constexpr uint32_t acc_cb = (MODE == MODE_STATS_OUT) ? cb_out : cb_acc;
    constexpr uint32_t norm_cb = cb_out;

    constexpr uint32_t startup_out = (MODE == MODE_STATS_IN) ? norm_cb : acc_cb;  // a CB this build allocates
    compute_kernel_hw_startup(cb_x, cb_x, startup_out);
    if constexpr (MODE != MODE_STATS_OUT) {
        cb_wait_front(cb_scaler, 1);  // from the reader, once
    }

    for (uint32_t u = 0; u < num_units; ++u) {
        if constexpr (MODE != MODE_STATS_IN) {
            // ---- pass A: sum over the unit's tiles of x^2, accumulated by the packer in fp32 L1
            reconfig_data_format(cb_x, cb_x);
            pack_reconfig_data_format(acc_cb);
            pack_reconfig_l1_acc(0);
            mul_init(cb_x, cb_x);
            cb_reserve_back(acc_cb, 1);
            for (uint32_t col = 0; col < WB; col += BLK) {
                cb_wait_front(cb_x, BLK);
                tile_regs_acquire();
                for (uint32_t i = 0; i < BLK; ++i) {
                    mul_tiles(cb_x, cb_x, i, i, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < BLK; ++i) {
                    pack_tile<true>(i, acc_cb, 0);
                    if (col == 0 && i == 0) {
                        pack_reconfig_l1_acc(1);  // the first tile overwrites, the rest accumulate
                    }
                }
                tile_regs_release();
                cb_pop_front(cb_x, BLK);
            }
            pack_reconfig_l1_acc(0);
            cb_push_back(acc_cb, 1);
        }
        if constexpr (MODE == MODE_STATS_OUT) {
            continue;
        }
        // ---- the row statistic: SUM over every column of the partial tile(s), then the SFPU finalize in DEST
        constexpr uint32_t src = (MODE == MODE_STATS_IN) ? cb_stats : cb_acc;
        constexpr uint32_t ntiles = (MODE == MODE_STATS_IN) ? G : 1;
        reconfig_data_format(cb_scaler, src);  // REDUCE_ROW SUM: scaler -> srcA, data -> srcB
        pack_reconfig_data_format(cb_rs);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(src, cb_scaler, cb_rs);
        cb_wait_front(src, ntiles);
        tile_regs_acquire();
        for (uint32_t t = 0; t < ntiles; ++t) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(src, cb_scaler, t, 0, 0);
        }
        binop_with_scalar_tile_init();
        mul_unary_tile(0, INV_W_BITS);
        add_unary_tile(0, EPS_BITS);
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();
        cb_pop_front(src, ntiles);
        cb_reserve_back(cb_rs, 1);
        tile_regs_wait();
        pack_tile(0, cb_rs);
        tile_regs_release();
        cb_push_back(cb_rs, 1);
        reduce_uninit();

        // ---- pass B: y = x * rs (FPU, column broadcast) into DEST, then [* gamma] on the SFPU in DEST (gamma
        // unpacked with its row broadcast; fp32 gamma goes through unpack-to-dest, exact): y is rounded once, on
        // the pack to the output's format (no fp32 L1 round trip between the two multiplies)
        cb_wait_front(cb_rs, 1);
        for (uint32_t col = 0; col < WB; col += BLK) {
            reconfig_data_format(cb_x, cb_rs);
            pack_reconfig_data_format(cb_out);
            mul_bcast_cols_init(cb_x, cb_rs);
            cb_wait_front(cb_x, BLK);
            if constexpr (HAS_GAMMA) {
                cb_wait_front(cb_gamma, BLK);
            }
            cb_reserve_back(cb_out, BLK);
            tile_regs_acquire();
            for (uint32_t i = 0; i < BLK; ++i) {
                mul_tiles_bcast_cols(cb_x, cb_rs, i, 0, i);
            }
            if constexpr (HAS_GAMMA) {
                reconfig_data_format(cb_gamma, cb_gamma);
                unary_bcast_init<BroadcastType::ROW>(cb_gamma);
                for (uint32_t i = 0; i < BLK; ++i) {
                    unary_bcast<BroadcastType::ROW>(cb_gamma, i, BLK + i);
                }
                unary_bcast_uninit<BroadcastType::ROW>(cb_gamma);
                mul_binary_tile_init();
                for (uint32_t i = 0; i < BLK; ++i) {
                    mul_binary_tile(i, BLK + i, i);
                }
            }
            if constexpr (HAS_TWIN) {
                // (the gamma tiles in BLK + i are spent: the rounded copy goes there)
                MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>()));
                for (uint32_t i = 0; i < BLK; ++i) {
                    MATH((_llk_math_eltwise_unary_sfpu_params_(rms_twin::round_bf16_away<BLK>, i, VectorMode::RC)));
                }
                cb_reserve_back(cb_twin, BLK);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < BLK; ++i) {
                pack_tile(i, cb_out);
            }
            if constexpr (HAS_TWIN) {
                pack_reconfig_data_format(cb_out, cb_twin);
                for (uint32_t i = 0; i < BLK; ++i) {
                    pack_tile(BLK + i, cb_twin);
                }
            }
            tile_regs_release();
            cb_pop_front(cb_x, BLK);
            if constexpr (HAS_GAMMA) {
                cb_pop_front(cb_gamma, BLK);
            }
            cb_push_back(cb_out, BLK);
            if constexpr (HAS_TWIN) {
                cb_push_back(cb_twin, BLK);
            }
        }
        cb_pop_front(cb_rs, 1);
    }
}
