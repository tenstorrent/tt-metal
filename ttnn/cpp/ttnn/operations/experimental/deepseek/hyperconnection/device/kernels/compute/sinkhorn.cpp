// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Decode-only (T == 1) comb / Sinkhorn-Knopp compute on a single [H,H] tile.
//
//   comb = softmax(comb_w * comb_scale + comb_bias, dim=-1) + eps
//   comb = comb / (sum(comb, dim=-2) + eps)                     (initial column pass)
//   repeat sinkhorn_iters-1 times: row pass (dim=-1) then column pass (dim=-2)
//
// The logical region is HxH inside a 32x32 tile. A ones-mask tile (1 inside HxH, 0 in the
// tile padding) is multiplied in right after the softmax exp so every reduction that follows
// only sees the valid HxH block. eps is added to every reciprocal denominator so that the
// all-zero padding rows/cols reduce to 0 (0 * 1/eps) instead of 0 * inf = NaN.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

namespace {

// logits = comb_w * scale + comb_bias  -> cb_comb.
void produce_logits(uint32_t cb_w, uint32_t cb_bias, uint32_t cb_scratch, uint32_t cb_comb, uint32_t scale_bits) {
    cb_wait_front(cb_w, 1);

    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_w);
    copy_tile(cb_w, 0, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, scale_bits);
    tile_regs_commit();

    cb_reserve_back(cb_scratch, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_scratch);
    pack_tile(0, cb_scratch);
    tile_regs_release();
    cb_push_back(cb_scratch, 1);
    cb_pop_front(cb_w, 1);

    cb_wait_front(cb_scratch, 1);
    cb_wait_front(cb_bias, 1);

    tile_regs_acquire();
    add_tiles_init(cb_scratch, cb_bias);
    add_tiles(cb_scratch, cb_bias, 0, 0, 0);
    tile_regs_commit();

    cb_pop_front(cb_scratch, 1);
    cb_pop_front(cb_bias, 1);

    cb_reserve_back(cb_comb, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_comb);
    pack_tile(0, cb_comb);
    tile_regs_release();
    cb_push_back(cb_comb, 1);
}

template <PoolType pool, ReduceDim dim>
void reduce_to_cb(uint32_t cb_in, uint32_t cb_scaler, uint32_t cb_out, bool eps_recip, uint32_t eps_bits) {
    cb_wait_front(cb_in, 1);

    reduce_init<pool, dim>(cb_in, cb_scaler, cb_out);
    tile_regs_acquire();
    reduce_tile<pool, dim>(cb_in, cb_scaler, 0, 0, 0);
    if (eps_recip) {
        binop_with_scalar_tile_init();
        add_unary_tile(0, eps_bits);
        recip_tile_init();
        recip_tile(0);
    }
    tile_regs_commit();
    reduce_uninit();

    cb_reserve_back(cb_out, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_out);
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}

// comb = exp(comb - rowmax), rowmax broadcast across columns (COL bcast).
void sub_max_exp(uint32_t cb_comb, uint32_t cb_red) {
    cb_wait_front(cb_comb, 1);
    cb_wait_front(cb_red, 1);

    sub_bcast_cols_init_short(cb_comb, cb_red);
    tile_regs_acquire();
    sub_tiles_bcast_cols(cb_comb, cb_red, 0, 0, 0);
    exp_tile_init();
    exp_tile(0);
    tile_regs_commit();

    cb_pop_front(cb_comb, 1);
    cb_pop_front(cb_red, 1);

    cb_reserve_back(cb_comb, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_comb);
    pack_tile(0, cb_comb);
    tile_regs_release();
    cb_push_back(cb_comb, 1);
}

// comb *= mask (elementwise). mask is reused, so it is not popped.
void mul_mask(uint32_t cb_comb, uint32_t cb_mask) {
    cb_wait_front(cb_comb, 1);
    cb_wait_front(cb_mask, 1);

    mul_tiles_init(cb_comb, cb_mask);
    tile_regs_acquire();
    mul_tiles(cb_comb, cb_mask, 0, 0, 0);
    tile_regs_commit();

    cb_pop_front(cb_comb, 1);

    cb_reserve_back(cb_comb, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_comb);
    pack_tile(0, cb_comb);
    tile_regs_release();
    cb_push_back(cb_comb, 1);
}

// comb += eps (elementwise).
void add_eps(uint32_t cb_comb, uint32_t eps_bits) {
    cb_wait_front(cb_comb, 1);

    copy_tile_to_dst_init_short(cb_comb);
    tile_regs_acquire();
    copy_tile(cb_comb, 0, 0);
    binop_with_scalar_tile_init();
    add_unary_tile(0, eps_bits);
    tile_regs_commit();

    cb_pop_front(cb_comb, 1);

    cb_reserve_back(cb_comb, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_comb);
    pack_tile(0, cb_comb);
    tile_regs_release();
    cb_push_back(cb_comb, 1);
}

// Multiply comb by a broadcast reciprocal vector. is_col=true broadcasts a [H,1] column across
// W (row normalization); is_col=false broadcasts a [1,W] row across H (column normalization).
void mul_bcast_recip(uint32_t cb_comb, uint32_t cb_red, bool is_col) {
    cb_wait_front(cb_comb, 1);
    cb_wait_front(cb_red, 1);

    if (is_col) {
        mul_bcast_cols_init_short(cb_comb, cb_red);
    } else {
        mul_bcast_rows_init_short(cb_comb, cb_red);
    }
    tile_regs_acquire();
    if (is_col) {
        mul_tiles_bcast_cols(cb_comb, cb_red, 0, 0, 0);
    } else {
        mul_tiles_bcast_rows(cb_comb, cb_red, 0, 0, 0);
    }
    tile_regs_commit();

    cb_pop_front(cb_comb, 1);
    cb_pop_front(cb_red, 1);

    cb_reserve_back(cb_comb, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_comb);
    pack_tile(0, cb_comb);
    tile_regs_release();
    cb_push_back(cb_comb, 1);
}

void copy_to_out(uint32_t cb_comb, uint32_t cb_out) {
    cb_wait_front(cb_comb, 1);

    copy_tile_to_dst_init_short(cb_comb);
    tile_regs_acquire();
    copy_tile(cb_comb, 0, 0);
    tile_regs_commit();

    cb_pop_front(cb_comb, 1);

    cb_reserve_back(cb_out, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_out);
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_comb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_comb_bias = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(2);
    constexpr uint32_t cb_mask = get_compile_time_arg_val(3);
    constexpr uint32_t cb_comb = get_compile_time_arg_val(4);
    constexpr uint32_t cb_reduce = get_compile_time_arg_val(5);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(6);
    constexpr uint32_t cb_out = get_compile_time_arg_val(7);
    constexpr uint32_t num_streams = get_compile_time_arg_val(8);  // unused at runtime (masking via cb_mask)
    constexpr uint32_t sinkhorn_iters = get_compile_time_arg_val(9);
    constexpr uint32_t comb_scale_bits = get_compile_time_arg_val(10);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(11);
    (void)num_streams;

    binary_op_init_common(cb_comb_w, cb_comb_bias, cb_comb);

    cb_wait_front(cb_scaler, 1);

    // logits = comb_w * comb_scale + comb_bias.
    produce_logits(cb_comb_w, cb_comb_bias, cb_scratch, cb_comb, comb_scale_bits);

    // softmax over the last dim (W), masked to the valid HxH block.
    reduce_to_cb<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_comb, cb_scaler, cb_reduce, /*eps_recip=*/false, eps_bits);
    sub_max_exp(cb_comb, cb_reduce);
    mul_mask(cb_comb, cb_mask);
    reduce_to_cb<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_comb, cb_scaler, cb_reduce, /*eps_recip=*/true, eps_bits);
    mul_bcast_recip(cb_comb, cb_reduce, /*is_col=*/true);
    add_eps(cb_comb, eps_bits);
    mul_mask(cb_comb, cb_mask);

    // initial column normalisation: comb /= (sum_rows(comb) + eps).
    reduce_to_cb<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_comb, cb_scaler, cb_reduce, /*eps_recip=*/true, eps_bits);
    mul_bcast_recip(cb_comb, cb_reduce, /*is_col=*/false);

    // Sinkhorn-Knopp iterations: alternate row then column normalisation.
    for (uint32_t i = 1; i < sinkhorn_iters; ++i) {
        reduce_to_cb<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_comb, cb_scaler, cb_reduce, /*eps_recip=*/true, eps_bits);
        mul_bcast_recip(cb_comb, cb_reduce, /*is_col=*/true);
        reduce_to_cb<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_comb, cb_scaler, cb_reduce, /*eps_recip=*/true, eps_bits);
        mul_bcast_recip(cb_comb, cb_reduce, /*is_col=*/false);
    }

    copy_to_out(cb_comb, cb_out);
}
