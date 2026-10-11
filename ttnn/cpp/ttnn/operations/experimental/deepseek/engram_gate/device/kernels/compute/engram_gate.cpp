// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "api/compute/transpose.h"

// Engram gate on one D-slice. A row block is 32 (user, copy) rows; per block:
//   phase 1: partial[q] = transpose(row_sum(term_q)), term = x^2, key^2, x * gate_weight * key
//            -> the writer scatters row 0 of each partial into row c of every core's gather tile
//   phase 2: S[q] = col_sum(gather[q])            (full-D sums, one value per row, in row 0)
//            gate = sigmoid(sign(dot) * sqrt(max(|dot|, clamp))),
//            dot  = S2 * rsqrt(S0 / D + eps) * rsqrt(S1 / D + eps)
//            out  = x + transpose(gate) (col-broadcast) * value      -> untilized into cb_out_rm
// Every tile CB is fp32; only the row-major inputs and the output are bf16.
namespace {

void tilize_block_rows(uint32_t cb_rm, uint32_t Wt, uint32_t block, uint32_t cb_tiles) {
    reconfig_data_format_srca(cb_rm);
    tilize_init(cb_rm, Wt, cb_tiles);
    cb_reserve_back(cb_tiles, Wt);
    tilize_block(cb_rm, Wt, cb_tiles, block * Wt);
    cb_push_back(cb_tiles, Wt);
    tilize_uninit(cb_rm, cb_tiles);
    reconfig_data_format_srca(cb_tiles);
}

// cb_prod (Wt tiles) -> row sums in column 0 -> transposed into row 0 of one cb_partial tile.
void row_sum_to_partial(uint32_t cb_prod, uint32_t Wt, uint32_t cb_scaler, uint32_t cb_col, uint32_t cb_partial) {
    cb_wait_front(cb_prod, Wt);
    cb_reserve_back(cb_col, 1);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_prod, cb_scaler, cb_col);
    tile_regs_acquire();
    for (uint32_t wt = 0; wt < Wt; ++wt) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_prod, cb_scaler, wt, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_col);
    tile_regs_release();
    reduce_uninit(cb_prod);
    cb_pop_front(cb_prod, Wt);
    cb_push_back(cb_col, 1);

    cb_wait_front(cb_col, 1);
    cb_reserve_back(cb_partial, 1);
    transpose_init(cb_col);
    tile_regs_acquire();
    transpose_tile(cb_col, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_partial);
    tile_regs_release();
    cb_pop_front(cb_col, 1);
    cb_push_back(cb_partial, 1);
}

void square_into(uint32_t cb_in, uint32_t Wt, uint32_t cb_prod) {
    cb_reserve_back(cb_prod, Wt);
    mul_init(cb_in, cb_in);
    for (uint32_t wt = 0; wt < Wt; ++wt) {
        tile_regs_acquire();
        mul_tiles(cb_in, cb_in, wt, wt, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_prod);
        tile_regs_release();
    }
    cb_push_back(cb_prod, Wt);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_x_rm = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k_rm = get_compile_time_arg_val(1);
    constexpr uint32_t cb_v_rm = get_compile_time_arg_val(2);
    constexpr uint32_t cb_gate = get_compile_time_arg_val(3);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(4);
    constexpr uint32_t cb_xt = get_compile_time_arg_val(5);
    constexpr uint32_t cb_kt = get_compile_time_arg_val(6);
    constexpr uint32_t cb_vt = get_compile_time_arg_val(7);
    constexpr uint32_t cb_prod = get_compile_time_arg_val(8);
    constexpr uint32_t cb_col = get_compile_time_arg_val(9);
    constexpr uint32_t cb_partial = get_compile_time_arg_val(10);
    constexpr uint32_t cb_gather = get_compile_time_arg_val(11);
    constexpr uint32_t cb_stats = get_compile_time_arg_val(12);
    constexpr uint32_t cb_gate_row = get_compile_time_arg_val(13);
    constexpr uint32_t cb_gate_col = get_compile_time_arg_val(14);
    constexpr uint32_t cb_out_rm = get_compile_time_arg_val(15);
    constexpr uint32_t Rt = get_compile_time_arg_val(16);
    constexpr uint32_t Wt = get_compile_time_arg_val(17);
    constexpr uint32_t inv_d = get_compile_time_arg_val(18);
    constexpr uint32_t eps = get_compile_time_arg_val(19);
    constexpr uint32_t gate_clamp = get_compile_time_arg_val(20);
    constexpr uint32_t rm_pages = Rt * Wt;

    compute_kernel_hw_startup(cb_xt, cb_xt, cb_prod);

    cb_reserve_back(cb_gate, Wt);
    cb_push_back(cb_gate, Wt);
    cb_wait_front(cb_gate, Wt);
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_x_rm, rm_pages);
    cb_wait_front(cb_k_rm, rm_pages);

    // Phase 1: per-core partial sums.
    for (uint32_t rt = 0; rt < Rt; ++rt) {
        tilize_block_rows(cb_x_rm, Wt, rt, cb_xt);
        tilize_block_rows(cb_k_rm, Wt, rt, cb_kt);
        cb_wait_front(cb_xt, Wt);
        cb_wait_front(cb_kt, Wt);

        square_into(cb_xt, Wt, cb_prod);
        row_sum_to_partial(cb_prod, Wt, cb_scaler, cb_col, cb_partial);
        square_into(cb_kt, Wt, cb_prod);
        row_sum_to_partial(cb_prod, Wt, cb_scaler, cb_col, cb_partial);

        cb_reserve_back(cb_prod, Wt);
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            mul_init(cb_xt, cb_gate);
            tile_regs_acquire();
            mul_tiles(cb_xt, cb_gate, wt, wt, 0);
            mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_kt);
            mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_kt, wt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_prod);
            tile_regs_release();
        }
        cb_push_back(cb_prod, Wt);
        row_sum_to_partial(cb_prod, Wt, cb_scaler, cb_col, cb_partial);

        cb_pop_front(cb_xt, Wt);
        cb_pop_front(cb_kt, Wt);
    }
    cb_pop_front(cb_k_rm, rm_pages);

    // Phase 2: full sums, gate, residual.
    cb_wait_front(cb_v_rm, rm_pages);
    cb_wait_front(cb_gather, 3 * Rt);
    for (uint32_t rt = 0; rt < Rt; ++rt) {
        cb_reserve_back(cb_stats, 3);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_gather, cb_scaler, cb_stats);
        tile_regs_acquire();
        for (uint32_t q = 0; q < 3; ++q) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_gather, cb_scaler, 3 * rt + q, 0, q);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t q = 0; q < 3; ++q) {
            pack_tile(q, cb_stats);
        }
        tile_regs_release();
        reduce_uninit(cb_gather);
        cb_push_back(cb_stats, 3);

        cb_wait_front(cb_stats, 3);
        cb_reserve_back(cb_gate_row, 1);
        copy_init(cb_stats);
        tile_regs_acquire();
        copy_tile(cb_stats, 0, 0);
        copy_tile(cb_stats, 1, 1);
        copy_tile(cb_stats, 2, 2);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_d);
        add_unary_tile(0, eps);
        mul_unary_tile(1, inv_d);
        add_unary_tile(1, eps);
        rsqrt_tile_init();
        rsqrt_tile(0);
        rsqrt_tile(1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        mul_binary_tile(0, 2, 0);  // dot
        // rstd > 0, so sign(dot) == sign(S2).
        copy_init(cb_stats);
        copy_tile(cb_stats, 2, 1);
        sign_tile_init();
        sign_tile(1);
        abs_tile_init();
        abs_tile(0);
        unary_max_tile_init();
        unary_max_tile(0, gate_clamp);
        sqrt_tile_init();
        sqrt_tile(0);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        sigmoid_tile_init();
        sigmoid_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_gate_row);
        tile_regs_release();
        cb_pop_front(cb_stats, 3);
        cb_push_back(cb_gate_row, 1);

        cb_wait_front(cb_gate_row, 1);
        cb_reserve_back(cb_gate_col, 1);
        transpose_init(cb_gate_row);
        tile_regs_acquire();
        transpose_tile(cb_gate_row, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_gate_col);
        tile_regs_release();
        cb_pop_front(cb_gate_row, 1);
        cb_push_back(cb_gate_col, 1);

        tilize_block_rows(cb_x_rm, Wt, rt, cb_xt);
        tilize_block_rows(cb_v_rm, Wt, rt, cb_vt);
        cb_wait_front(cb_xt, Wt);
        cb_wait_front(cb_vt, Wt);
        cb_wait_front(cb_gate_col, 1);

        pack_untilize_dest_init<1, Wt>(cb_out_rm);
        cb_reserve_back(cb_out_rm, Wt);
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            mul_bcast_cols_init(cb_vt, cb_gate_col);
            tile_regs_acquire();
            mul_tiles_bcast_cols(cb_vt, cb_gate_col, wt, 0, 0);
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_xt);
            add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_xt, wt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dest<1, Wt>(cb_out_rm, 1, wt);
            tile_regs_release();
        }
        cb_push_back(cb_out_rm, Wt);
        pack_untilize_uninit(cb_out_rm);
        pack_reconfig_data_format(cb_xt);

        cb_pop_front(cb_xt, Wt);
        cb_pop_front(cb_vt, Wt);
        cb_pop_front(cb_gate_col, 1);
    }
    cb_pop_front(cb_gather, 3 * Rt);
    cb_pop_front(cb_x_rm, rm_pages);
    cb_pop_front(cb_v_rm, rm_pages);
}
