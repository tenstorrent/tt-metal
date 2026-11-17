// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Row core compute. Per tile-row of my block (Wt tiles of slice p):
//   S      = sum_j x_j * x_j                      (FPU, DEST-accumulated, fp32)     -> CB_PART (writer exchanges it)
//   xg_j   = x_j * gamma_j (gamma row-broadcast)  (overlaps the exchange)           -> CB_XG
//   inv    = rsqrt(rowsum(S_0 + .. + S_G-1) / W + eps)  (FPU row reduce of the G gathered S tiles, accumulated in
//            fp32 DEST in slice order; then one SFPU pass over column 0)           -> CB_INV
//   out_j  = xg_j * inv (column-broadcast)                                          -> CB_OUT

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "agrn_common.hpp"
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_converter.h"
#endif

using namespace agrn;

// rsqrt(x * (1/W) + eps) on column 0 of a DEST tile (the row sums of a REDUCE_ROW). Column 0 lives only in the
// even-parity SFPU vectors of faces 0 and 2: VectorMode::C visits faces 0 and 2, the body every other vector.
#ifdef TRISC_MATH
sfpi_inline void finalize_col0(const uint32_t inv_w_bits, const uint32_t eps_bits) {
#pragma GCC unroll 0
    for (int rg = 0; rg < 4; rg++) {
        const sfpi::vFloat inv_w = ckernel::sfpu::Converter::as_float(inv_w_bits);
        const sfpi::vFloat eps = ckernel::sfpu::Converter::as_float(eps_bits);
        sfpi::vFloat y =
            ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false>(sfpi::dst_reg[0] * inv_w + eps);
        if constexpr (!DST_ACCUM_MODE) {
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = y;
        sfpi::dst_reg += 2;
    }
}
#endif

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t rows = get_compile_time_arg_val(1);
    constexpr uint32_t G = get_compile_time_arg_val(2);
    constexpr uint32_t inv_w_bits = get_compile_time_arg_val(3);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(4);
    constexpr uint32_t dest_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t half_a = get_compile_time_arg_val(6);
    constexpr uint32_t P = get_compile_time_arg_val(7);
    using RL = RowLayout<Wt, half_a, P>;
    constexpr uint32_t blk = RL::slots * rows;

    const uint32_t ctr_addr = get_arg_val<uint32_t>(0);
    const uint32_t parity = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctr_addr) & 1;
    const uint32_t base = parity * blk;
    compute_kernel_hw_startup(CB_LAND, CB_LAND, CB_PART);
    {
        AGRN_ZONE("rc_wait_x");
        cb_wait_front(CB_LAND, rows);  // the reader publishes one page per tile-row once all of it is here
    }
    cb_wait_front(CB_SCALER, 1);
    cb_wait_front(CB_GAMMA, Wt);

    for (uint32_t r = 0; r < rows; ++r) {
        const uint32_t rb = base + r * RL::slots;
        {
            AGRN_ZONE("rc_front");
            // ---- S = sum_j x_j^2 (fp32, DEST-accumulated) -> CB_PART: the writer sends it to every core of my row
            // ----
            reconfig_data_format(CB_LAND, CB_LAND);
            pack_reconfig_data_format(CB_PART);
            mul_init(CB_LAND, CB_LAND, true);
            cb_reserve_back(CB_PART, 1);
            tile_regs_acquire();
            for (uint32_t j = 0; j < Wt; ++j) {
                mul_tiles(CB_LAND, CB_LAND, rb + RL::slot_of(j), rb + RL::slot_of(j), 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_PART);
            tile_regs_release();
            cb_push_back(CB_PART, 1);

            // ---- xg_j = x_j * gamma_j ----
            reconfig_data_format(CB_LAND, CB_GAMMA);
            pack_reconfig_data_format(CB_XG);
            mul_bcast_rows_init(CB_LAND, CB_GAMMA);
            for (uint32_t j0 = 0; j0 < Wt; j0 += dest_tiles) {
                const uint32_t n = (Wt - j0) < dest_tiles ? (Wt - j0) : dest_tiles;
                cb_reserve_back(CB_XG, n);
                tile_regs_acquire();
                for (uint32_t k = 0; k < n; ++k) {
                    mul_tiles_bcast_rows(CB_LAND, CB_GAMMA, rb + RL::slot_of(j0 + k), j0 + k, k);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t k = 0; k < n; ++k) {
                    pack_tile(k, CB_XG);
                }
                tile_regs_release();
                cb_push_back(CB_XG, n);
            }
        }
        {
            AGRN_ZONE("rc_inv");
            // ---- inv = rsqrt(sum_p part_p / W + eps) (SFPU, fp32) ----
            {
                AGRN_ZONE("rc_wait_gather");
                cb_wait_front(CB_GATHER, G);
            }
            // row sums of the G S-tiles, accumulated in fp32 DEST in slice order (the same order on every core and
            // chip), then rsqrt(sum / W + eps) on column 0 only
            reconfig_data_format(CB_SCALER, CB_GATHER);
            pack_reconfig_data_format(CB_INV);
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_GATHER, CB_SCALER, CB_INV);
            cb_reserve_back(CB_INV, 1);
            tile_regs_acquire();
            for (uint32_t k = 0; k < G; ++k) {
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_GATHER, CB_SCALER, k, 0, 0);
            }
            rsqrt_tile_init();
            MATH((_llk_math_eltwise_unary_sfpu_params_(finalize_col0, 0, VectorMode::C, inv_w_bits, eps_bits)));
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_INV);
            tile_regs_release();
            cb_push_back(CB_INV, 1);
            reduce_uninit();
            cb_pop_front(CB_GATHER, G);
        }
        AGRN_ZONE("rc_out");
        // ---- out_j = xg_j * inv ----
        cb_wait_front(CB_INV, 1);
        cb_wait_front(CB_XG, Wt);
        reconfig_data_format(CB_XG, CB_INV);
        pack_reconfig_data_format(CB_OUT);
        mul_bcast_cols_init(CB_XG, CB_INV);
        for (uint32_t j0 = 0; j0 < Wt; j0 += dest_tiles) {
            const uint32_t n = (Wt - j0) < dest_tiles ? (Wt - j0) : dest_tiles;
            cb_reserve_back(CB_OUT, n);
            tile_regs_acquire();
            for (uint32_t k = 0; k < n; ++k) {
                mul_tiles_bcast_cols(CB_XG, CB_INV, j0 + k, 0, k);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t k = 0; k < n; ++k) {
                pack_tile(k, CB_OUT);
            }
            tile_regs_release();
            cb_push_back(CB_OUT, n);
        }
        cb_pop_front(CB_XG, Wt);
        cb_pop_front(CB_INV, 1);
    }
    cb_pop_front(CB_GAMMA, Wt);
    cb_pop_front(CB_SCALER, 1);
    cb_pop_front(CB_LAND, rows);
}
