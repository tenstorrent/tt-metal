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
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"

// Engram gate for one (32-token block, stream copy), over the full D on this core.
//
//   S0 = sum_d x^2     S1 = sum_d key^2     S2 = sum_d x * gate * key
//   rstd = rsqrt(S0/D + eps) * rsqrt(S1/D + eps)
//   dot  = S2 * rstd
//   gate = sigmoid(sign(dot) * sqrt(max(|dot|, clamp)))   (rstd > 0, so sign(dot) == sign(S2))
//   out  = x + gate * value
//
// D is walked in chunks, because a chunk is all a core holds. Phase 1 leaves one partial tile per chunk per
// statistic and reduces them at the end; the gate is then formed once and broadcast down value's columns.
//
// x, key, value and the output are bf16 CBs; every other CB is fp32. The kernel switches the unpack/pack formats
// explicitly whenever it moves between the two, since the op inits don't.
namespace {

// cb_prod (Wt tiles) -> row sums in column 0, transposed into row 0 of cb_partial.
void row_sum_to_partial(uint32_t cb_prod, uint32_t Wt, uint32_t cb_scaler, uint32_t cb_col, uint32_t cb_partial) {
    cb_wait_front(cb_prod, Wt);
    cb_reserve_back(cb_col, 1);
    reconfig_data_format(cb_prod, cb_scaler);
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

// Squares the Wt tiles of cb_in into cb_prod.
void square_into(uint32_t cb_in, uint32_t Wt, uint32_t cb_prod) {
    reconfig_data_format(cb_in, cb_in);
    pack_reconfig_data_format(cb_prod);
    mul_init(cb_in, cb_in);
    cb_reserve_back(cb_prod, Wt);
    for (uint32_t wt = 0; wt < Wt; ++wt) {
        tile_regs_acquire();
        mul_tiles(cb_in, cb_in, wt, wt, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_prod, wt);
        tile_regs_release();
    }
    cb_push_back(cb_prod, Wt);
}

// Adds the `n` partial tiles of cb_partial into one, packed into cb_stats.
void sum_partials(uint32_t cb_partial, uint32_t n, uint32_t cb_stats) {
    cb_wait_front(cb_partial, n);
    cb_reserve_back(cb_stats, 1);
    copy_init(cb_partial);
    tile_regs_acquire();
    copy_tile(cb_partial, 0, 0);
    add_binary_tile_init();
    for (uint32_t i = 1; i < n; ++i) {
        copy_tile(cb_partial, i, 1);
        add_binary_tile(0, 1, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_stats);
    tile_regs_release();
    cb_pop_front(cb_partial, n);
    cb_push_back(cb_stats, 1);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k = get_compile_time_arg_val(1);
    constexpr uint32_t cb_g = get_compile_time_arg_val(2);
    constexpr uint32_t cb_v = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    constexpr uint32_t cb_prod = get_compile_time_arg_val(5);
    constexpr uint32_t cb_p0 = get_compile_time_arg_val(6);
    constexpr uint32_t cb_p1 = get_compile_time_arg_val(7);
    constexpr uint32_t cb_p2 = get_compile_time_arg_val(8);
    constexpr uint32_t cb_stats = get_compile_time_arg_val(9);
    constexpr uint32_t cb_col = get_compile_time_arg_val(10);
    constexpr uint32_t cb_gate_row = get_compile_time_arg_val(11);
    constexpr uint32_t cb_gate_col = get_compile_time_arg_val(12);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(13);
    constexpr uint32_t Wt = get_compile_time_arg_val(14);  // tiles per chunk
    constexpr uint32_t num_chunks = get_compile_time_arg_val(15);
    constexpr uint32_t inv_d = get_compile_time_arg_val(16);
    constexpr uint32_t eps = get_compile_time_arg_val(17);
    constexpr uint32_t gate_clamp = get_compile_time_arg_val(18);

    const uint32_t unit_start = get_arg_val<uint32_t>(0);
    const uint32_t units = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_x, cb_x, cb_prod);
    cb_wait_front(cb_scaler, 1);

    for (uint32_t u = unit_start; u < unit_start + units; ++u) {
        // ---- phase 1: a partial row sum per chunk, per statistic ----
        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            cb_wait_front(cb_x, Wt);
            cb_wait_front(cb_k, Wt);
            cb_wait_front(cb_g, Wt);

            square_into(cb_x, Wt, cb_prod);
            row_sum_to_partial(cb_prod, Wt, cb_scaler, cb_col, cb_p0);
            square_into(cb_k, Wt, cb_prod);
            row_sum_to_partial(cb_prod, Wt, cb_scaler, cb_col, cb_p1);

            // x * gate lands in srcB, which keeps gate's fp32 format; key shares x's bf16 srcA format.
            reconfig_data_format(cb_x, cb_g);
            cb_reserve_back(cb_prod, Wt);
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                // The dest-reuse init below reprograms the unpack and math MOPs, so the two-operand op must be
                // re-initialised every tile or the threads disagree on how many source handoffs a tile takes.
                mul_init(cb_x, cb_g);
                tile_regs_acquire();
                mul_tiles(cb_x, cb_g, wt, wt, 0);
                mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_k);
                mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_k, wt, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_prod, wt);
                tile_regs_release();
            }
            cb_push_back(cb_prod, Wt);
            row_sum_to_partial(cb_prod, Wt, cb_scaler, cb_col, cb_p2);
            cb_pop_front(cb_x, Wt);
            cb_pop_front(cb_k, Wt);
            cb_pop_front(cb_g, Wt);
        }

        // ---- the gate: reduce the chunks, then rsqrt / sign / sqrt / sigmoid ----
        sum_partials(cb_p0, num_chunks, cb_stats);  // S0
        sum_partials(cb_p1, num_chunks, cb_stats);  // S1
        sum_partials(cb_p2, num_chunks, cb_stats);  // S2
        cb_wait_front(cb_stats, 3);

        cb_reserve_back(cb_gate_row, 1);
        reconfig_data_format_srca(cb_stats);
        copy_init(cb_stats);
        tile_regs_acquire();
        copy_tile(cb_stats, 0, 0);  // S0
        copy_tile(cb_stats, 1, 1);  // S1
        copy_tile(cb_stats, 2, 2);  // S2
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_d);
        add_unary_tile(0, eps);
        mul_unary_tile(1, inv_d);
        add_unary_tile(1, eps);
        rsqrt_tile_init();
        rsqrt_tile(0);
        rsqrt_tile(1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);  // rstd
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

        // One gate value per row: transpose it to broadcast down each output tile's columns.
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

        // ---- phase 2: out = x + gate * value ----
        cb_wait_front(cb_gate_col, 1);
        // value and x share the bf16 srcA format; srcB holds the fp32 gate, then gate * value from dest.
        reconfig_data_format(cb_v, cb_gate_col);
        pack_reconfig_data_format(cb_out);
        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            cb_wait_front(cb_x, Wt);
            cb_wait_front(cb_v, Wt);
            cb_reserve_back(cb_out, Wt);
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                mul_bcast_cols_init(cb_v, cb_gate_col);
                tile_regs_acquire();
                mul_tiles_bcast_cols(cb_v, cb_gate_col, wt, 0, 0);
                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_x);
                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_x, wt, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_out, wt);
                tile_regs_release();
            }
            cb_push_back(cb_out, Wt);
            cb_pop_front(cb_x, Wt);
            cb_pop_front(cb_v, Wt);
        }
        cb_pop_front(cb_gate_col, 1);
    }
}
