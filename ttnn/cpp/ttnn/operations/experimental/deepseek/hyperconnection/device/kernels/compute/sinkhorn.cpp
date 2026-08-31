// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// comb / Sinkhorn-Knopp compute, one independent [H,H] tile per token.
//
//   comb = softmax(comb_w * comb_scale + comb_bias, dim=-1) + eps
//   comb = comb / (sum(comb, dim=-2) + eps)                     (initial column pass)
//   repeat sinkhorn_iters-1 times: row pass (dim=-1) then column pass (dim=-2)
//
// Hybrid layout: the initial phase (softmax -> masked `+= eps` -> first column normalization)
// runs through the tt-metal reduce/bcast API because the masked `+= eps` step cannot be
// expressed inside the SFPU sinkhorn_4x4 fast path (it needs the eps-mask tile). The
// remaining sinkhorn_iters-1 row+col passes are handed to the Blackhole SFPU fast path
// `sinkhorn_4x4<REMAINING_ITERS>`, which keeps the tile in fp32 LREGs across every
// row+col round and packs bf16 exactly once per token. Newton-refined SFPARECIP inside
// sinkhorn_4x4 (see ckernel_sfpu_sinkhorn.h) keeps the reciprocals at ~15+ bit precision.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/experimental/sinkhorn.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

namespace {

void produce_logits(uint32_t cb_w, uint32_t cb_bias, uint32_t cb_comb, uint32_t scale_bits) {
    cb_wait_front(cb_w, 1);
    cb_wait_front(cb_bias, 1);

    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_w);
    copy_tile(cb_w, 0, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, scale_bits);
    copy_tile_to_dst_init_short(cb_bias);
    copy_tile(cb_bias, 0, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    tile_regs_commit();

    cb_pop_front(cb_w, 1);

    cb_reserve_back(cb_comb, 1);
    tile_regs_wait();
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

void sub_max_exp_mask(uint32_t cb_comb, uint32_t cb_red, uint32_t cb_mask) {
    cb_wait_front(cb_comb, 1);
    cb_wait_front(cb_red, 1);
    cb_wait_front(cb_mask, 1);

    sub_bcast_cols_init(cb_comb, cb_red);
    tile_regs_acquire();
    sub_tiles_bcast_cols(cb_comb, cb_red, 0, 0, 0);
    exp_tile_init();
    exp_tile(0);
    copy_tile_to_dst_init_short(cb_mask);
    copy_tile(cb_mask, 0, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    tile_regs_commit();

    cb_pop_front(cb_comb, 1);
    cb_pop_front(cb_red, 1);

    cb_reserve_back(cb_comb, 1);
    tile_regs_wait();
    pack_tile(0, cb_comb);
    tile_regs_release();
    cb_push_back(cb_comb, 1);
}

void mul_bcast_recip(uint32_t cb_comb, uint32_t cb_red, bool is_col, uint32_t cb_eps_mask = 0xFFFFFFFF) {
    cb_wait_front(cb_comb, 1);
    cb_wait_front(cb_red, 1);

    if (is_col) {
        mul_bcast_cols_init(cb_comb, cb_red);
    } else {
        mul_bcast_rows_init(cb_comb, cb_red);
    }
    tile_regs_acquire();
    if (is_col) {
        mul_tiles_bcast_cols(cb_comb, cb_red, 0, 0, 0);
    } else {
        mul_tiles_bcast_rows(cb_comb, cb_red, 0, 0, 0);
    }
    if (cb_eps_mask != 0xFFFFFFFF) {
        cb_wait_front(cb_eps_mask, 1);
        copy_tile_to_dst_init_short(cb_eps_mask);
        copy_tile(cb_eps_mask, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    }
    tile_regs_commit();

    cb_pop_front(cb_comb, 1);
    cb_pop_front(cb_red, 1);

    cb_reserve_back(cb_comb, 1);
    tile_regs_wait();
    pack_tile(0, cb_comb);
    tile_regs_release();
    cb_push_back(cb_comb, 1);
}

// Run all remaining Sinkhorn-Knopp iterations through the SFPU fast path in one shot:
// load comb into DEST fp32 LREGs, do REMAINING_ITERS row+col passes without leaving LREGs,
// pack to bf16 once at the end. Every intermediate value stays fp32; only the final output
// is quantized. See ckernel_sfpu_sinkhorn.h for the LLK details.
template <
    uint32_t NUM_FACES_USED,
    uint32_t REMAINING_ITERS,
    uint32_t EPS_BITS,
    bool SINGLE_SUBMAT,
    uint32_t VALID_H,
    uint32_t VALID_W>
void sinkhorn_tail_in_dest(uint32_t cb_comb, uint32_t cb_out) {
    cb_wait_front(cb_comb, 1);

    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_comb);
    copy_tile(cb_comb, 0, 0);
    ckernel::sinkhorn_4x4_init();
    ckernel::sinkhorn_4x4<
        /*NUM_FACES_USED=*/NUM_FACES_USED,
        /*ITERS=*/REMAINING_ITERS,
        /*EPS_BITS=*/EPS_BITS,
        /*SINGLE_SUBMAT=*/SINGLE_SUBMAT,
        /*VALID_H=*/VALID_H,
        /*VALID_W=*/VALID_W>(0);
    tile_regs_commit();

    cb_pop_front(cb_comb, 1);

    cb_reserve_back(cb_out, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_out);
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
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
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_comb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_comb_bias = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(2);
    constexpr uint32_t cb_mask = get_compile_time_arg_val(3);
    constexpr uint32_t cb_comb = get_compile_time_arg_val(4);
    constexpr uint32_t cb_reduce = get_compile_time_arg_val(5);
    constexpr uint32_t cb_eps_mask = get_compile_time_arg_val(6);
    constexpr uint32_t cb_out = get_compile_time_arg_val(7);
    constexpr uint32_t num_streams = get_compile_time_arg_val(8);
    constexpr uint32_t sinkhorn_iters = get_compile_time_arg_val(9);
    constexpr uint32_t comb_scale_bits = get_compile_time_arg_val(10);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(11);

    constexpr uint32_t kValidH = num_streams;
    constexpr uint32_t kValidW = num_streams;
    constexpr uint32_t kNumFacesUsed = (num_streams <= 16) ? 1u : 4u;
    constexpr bool kSingleSubmat = (num_streams <= 4);
    constexpr uint32_t kRemainingIters = sinkhorn_iters - 1;

    compute_kernel_hw_startup(cb_comb_w, cb_comb_bias, cb_comb);

    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_comb_bias, 1);

    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        produce_logits(cb_comb_w, cb_comb_bias, cb_comb, comb_scale_bits);

        // Initial phase (reduce path -- bit-for-bit reference match).
        reduce_to_cb<PoolType::MAX, ReduceDim::REDUCE_ROW>(
            cb_comb, cb_scaler, cb_reduce, /*eps_recip=*/false, eps_bits);
        sub_max_exp_mask(cb_comb, cb_reduce, cb_mask);
        reduce_to_cb<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_comb, cb_scaler, cb_reduce, /*eps_recip=*/true, eps_bits);
        mul_bcast_recip(cb_comb, cb_reduce, /*is_col=*/true, cb_eps_mask);
        reduce_to_cb<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_comb, cb_scaler, cb_reduce, /*eps_recip=*/true, eps_bits);
        mul_bcast_recip(cb_comb, cb_reduce, /*is_col=*/false);

        // Remaining iterations: SFPU fast path, all inside one DEST cycle.
        if constexpr (kRemainingIters > 0) {
            sinkhorn_tail_in_dest<kNumFacesUsed, kRemainingIters, eps_bits, kSingleSubmat, kValidH, kValidW>(
                cb_comb, cb_out);
        } else {
            copy_to_out(cb_comb, cb_out);
        }
    }
}
