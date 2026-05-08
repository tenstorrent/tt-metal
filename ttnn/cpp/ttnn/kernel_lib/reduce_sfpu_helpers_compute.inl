// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Implementation file for reduce_sfpu_helpers_compute.hpp.
// Do not include directly -- include reduce_sfpu_helpers_compute.hpp instead.

#include <cstdint>

#include "api/compute/binary_max_min.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#ifdef REDUCE_POST_MUL
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/typecast.h"
#endif
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

#ifdef TRISC_PACK
#include "llk_pack_api.h"
#endif

namespace compute_kernel_lib {

namespace detail {

// Packer reduce mask: sfpu_reduce setup does not configure it; match FPU reduce behavior.
template <ckernel::ReduceDim reduce_dim>
ALWI void sfpu_reduce_pack_mask_config() {
    PACK((llk_pack_reduce_mask_config<false /*untilize*/, reduce_dim>()));
}

ALWI void sfpu_reduce_pack_mask_clear() { PACK((llk_pack_reduce_mask_clear())); }

template <ckernel::PoolType pool_type, DataFormat format>
ALWI void sfpu_reduce_binary_fold_init() {
    static_assert(format == DataFormat::Int32, "reduce_sfpu binary fold: Int32 only");
    if constexpr (pool_type == ckernel::PoolType::MAX) {
        binary_max_int32_tile_init();
    } else {
        binary_min_int32_tile_init();
    }
}

template <ckernel::PoolType pool_type, DataFormat format>
ALWI void sfpu_reduce_binary_fold_tile(uint32_t a, uint32_t b, uint32_t out) {
    static_assert(format == DataFormat::Int32, "reduce_sfpu binary fold: Int32 only");
    if constexpr (pool_type == ckernel::PoolType::MAX) {
        binary_max_int32_tile(a, b, out);
    } else {
        binary_min_int32_tile(a, b, out);
    }
}

}  // namespace detail

// Per output tile: (1) optional cross-tile binary fold along reduce axis, (2) sfpu_reduce in DST.
// Binary fold and sfpu_reduce each reprogram SFPCONFIG; re-init before each step every iteration.

template <ckernel::PoolType pool_type, ckernel::ReduceDim reduce_dim, DataFormat format>
ALWI void reduce_sfpu(
    uint32_t input_cb_id,
    uint32_t scaler_cb_id,
    uint32_t output_cb_id,
    ReduceInputBlockShape input_block_shape,
    uint32_t post_mul_scaler_bits) {
#ifndef REDUCE_POST_MUL
    (void)post_mul_scaler_bits;
#endif
    static_assert(
        pool_type == ckernel::PoolType::MAX || pool_type == ckernel::PoolType::MIN,
        "reduce_sfpu: MAX or MIN only");
    static_assert(
        reduce_dim == ckernel::ReduceDim::REDUCE_ROW || reduce_dim == ckernel::ReduceDim::REDUCE_COL,
        "reduce_sfpu: REDUCE_ROW or REDUCE_COL only");
    static_assert(format == DataFormat::Int32, "reduce_sfpu: Int32 only");
    static_assert(
        !(pool_type == ckernel::PoolType::MIN && reduce_dim == ckernel::ReduceDim::REDUCE_ROW),
        "reduce_sfpu: MIN + REDUCE_ROW unsupported (LLK); host must launch reduce_sfpu_w_neg.cpp");

    constexpr uint32_t onetile = 1;

    // Two DST registers used per output tile:
    //   acc_dst  - running max/min (initialised from the first input tile)
    //   work_dst - holds each subsequent input tile while binary_*_int32_tile
    //              folds it into acc_dst element-wise.
    constexpr uint32_t acc_dst = 0;
    constexpr uint32_t work_dst = 1;

    const uint32_t Ht = input_block_shape.rows;
    const uint32_t Wt = input_block_shape.cols;
    const uint32_t NC = input_block_shape.batches;

    const uint32_t tiles_per_output = (reduce_dim == ckernel::ReduceDim::REDUCE_ROW) ? Wt : Ht;
    const bool needs_cross_tile_fold = tiles_per_output > 1;

    init_sfpu(input_cb_id, output_cb_id);
    copy_tile_to_dst_init_short(input_cb_id);

    cb_wait_front(scaler_cb_id, onetile);

    detail::sfpu_reduce_pack_mask_config<reduce_dim>();

    const uint32_t outer_count = (reduce_dim == ckernel::ReduceDim::REDUCE_ROW) ? Ht : Wt;

    for (uint32_t nc = 0; nc < NC; ++nc) {
        for (uint32_t i = 0; i < outer_count; ++i) {
            tile_regs_acquire();

            if (needs_cross_tile_fold) {
                detail::sfpu_reduce_binary_fold_init<pool_type, format>();
            }

            cb_wait_front(input_cb_id, onetile);
            copy_tile(input_cb_id, 0, acc_dst);
            cb_pop_front(input_cb_id, onetile);

            for (uint32_t k = 1; k < tiles_per_output; ++k) {
                cb_wait_front(input_cb_id, onetile);
                copy_tile(input_cb_id, 0, work_dst);
                detail::sfpu_reduce_binary_fold_tile<pool_type, format>(acc_dst, work_dst, acc_dst);
                cb_pop_front(input_cb_id, onetile);
            }

            sfpu_reduce_init<pool_type, format>();
            sfpu_reduce<pool_type, format, reduce_dim>(acc_dst, /*ct_dim=*/1, /*rt_dim=*/1);

#ifdef REDUCE_POST_MUL
            // sfpu_reduce leaves Int32 bits in DST; mul_unary_tile is fp32-only.
            // Cast Int32 -> fp32, multiply, then cast fp32 -> Int32 (truncates toward zero).
            typecast_tile_init<(uint32_t)DataFormat::Int32, (uint32_t)DataFormat::Float32>();
            typecast_tile<(uint32_t)DataFormat::Int32, (uint32_t)DataFormat::Float32>(acc_dst);
            binop_with_scalar_tile_init();
            mul_unary_tile(acc_dst, post_mul_scaler_bits);
            typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Int32>();
            typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Int32>(acc_dst);
#endif

            tile_regs_commit();

            cb_reserve_back(output_cb_id, onetile);
            tile_regs_wait();
            pack_tile(acc_dst, output_cb_id);
            tile_regs_release();
            cb_push_back(output_cb_id, onetile);
        }
    }

    detail::sfpu_reduce_pack_mask_clear();
}

}  // namespace compute_kernel_lib
