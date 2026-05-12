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
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

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

// Cross-tile MAX fold along the reduce axis. MIN is dispatched to reduce_sfpu_{h,w}_neg.cpp
// where it is computed as -MAX(-x), so only the MAX variant is needed here.
template <DataFormat format>
ALWI void sfpu_reduce_max_fold_init() {
    static_assert(format == DataFormat::Int32 || format == DataFormat::Float32, "reduce_sfpu max fold: Int32 or Float32");
    if constexpr (format == DataFormat::Int32) {
        binary_max_int32_tile_init();
    } else {
        binary_max_tile_init();
    }
}

template <DataFormat format>
ALWI void sfpu_reduce_max_fold_tile(uint32_t a, uint32_t b, uint32_t out) {
    static_assert(format == DataFormat::Int32 || format == DataFormat::Float32, "reduce_sfpu max fold: Int32 or Float32");
    if constexpr (format == DataFormat::Int32) {
        binary_max_int32_tile(a, b, out);
    } else {
        binary_max_tile(a, b, out);
    }
}

#ifdef REDUCE_POST_MUL
// Post-reduction scaling. sfpu_reduce ignores the scaler CB (LLK takes no scalar), so the user
// scalar (packed fp32 bits) is applied here. mul_unary_tile is fp32-only; Int32 is bracketed with
// typecasts (truncates toward zero on the way back).
template <DataFormat format>
ALWI void sfpu_post_mul_tile(uint32_t dst, uint32_t scaler_bits) {
    static_assert(format == DataFormat::Int32 || format == DataFormat::Float32, "sfpu_post_mul_tile: Int32 or Float32");
    if constexpr (format == DataFormat::Int32) {
        typecast_tile_init<(uint32_t)DataFormat::Int32, (uint32_t)DataFormat::Float32>();
        typecast_tile<(uint32_t)DataFormat::Int32, (uint32_t)DataFormat::Float32>(dst);
        binop_with_scalar_tile_init();
        mul_unary_tile(dst, scaler_bits);
        typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Int32>();
        typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Int32>(dst);
    } else {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst, scaler_bits);
    }
}
#endif

}  // namespace detail

// =============================================================================
// Main Reduce Function Implementation
// =============================================================================

template <ckernel::PoolType pool_type, ckernel::ReduceDim reduce_dim, DataFormat format>
ALWI void reduce_sfpu(
    uint32_t input_cb_id,
    uint32_t scaler_cb_id,
    uint32_t output_cb_id,
    ReduceInputBlockShape input_block_shape,
    uint32_t post_mul_scaler_bits) {
    // =============================================================================
    // Static Assertions (compile-time validation)
    // =============================================================================
    static_assert(pool_type == ckernel::PoolType::MAX, "reduce_sfpu: MAX only; MIN dispatches to reduce_sfpu_{h,w}_neg.cpp");
    static_assert(
        reduce_dim == ckernel::ReduceDim::REDUCE_ROW || reduce_dim == ckernel::ReduceDim::REDUCE_COL,
        "reduce_sfpu: REDUCE_ROW or REDUCE_COL only");
    static_assert(format == DataFormat::Int32 || format == DataFormat::Float32, "reduce_sfpu: Int32 or Float32 only");

#ifndef REDUCE_POST_MUL
    (void)post_mul_scaler_bits;
#endif
    constexpr uint32_t onetile = 1;

    constexpr uint32_t dst_idx = 0;
    constexpr uint32_t next_dst_idx = 1;

    const uint32_t Ht = input_block_shape.rows;
    const uint32_t Wt = input_block_shape.cols;
    const uint32_t NC = input_block_shape.batches;

    init_sfpu(input_cb_id, output_cb_id);
    copy_tile_to_dst_init_short(input_cb_id);

    cb_wait_front(scaler_cb_id, onetile);

    detail::sfpu_reduce_pack_mask_config<reduce_dim>();

    if constexpr (reduce_dim == ckernel::ReduceDim::REDUCE_COL) {
        // =================================================================
        // REDUCE_COL: H reduction - each column -> 1 output tile (Wt outputs per batch)
        // Need chunking due to DEST register limits
        // StreamingPolicy: Tiles arrive in N C W_skip H W_chunk order (chunked by chunk_size)
        // PreloadedPolicy: Tiles in row-major order, indexed as batch_offset + ht*stride + wt
        // =================================================================

        // Auto-detect chunk size from DEST register capacity
        // Both reader (dataflow) and compute kernels compute this identically via DEST_AUTO_LIMIT
        constexpr uint32_t chunk_size = DEST_AUTO_LIMIT;
        constexpr uint32_t col_work_dst = chunk_size;
        const bool needs_cross_tile_fold = Ht > 1;

        for (uint32_t nc = 0; nc < NC; ++nc) {
            for (uint32_t wt_base = 0; wt_base < Wt; wt_base += chunk_size) {
                const uint32_t chunk_end = (wt_base + chunk_size < Wt) ? (wt_base + chunk_size) : Wt;
                const uint32_t current_chunk = chunk_end - wt_base;

                tile_regs_acquire();

                if (needs_cross_tile_fold) {
                    detail::sfpu_reduce_max_fold_init<format>();
                }

                for (uint32_t ht = 0; ht < Ht; ++ht) {
                    for (uint32_t k = 0; k < current_chunk; ++k) {
                        cb_wait_front(input_cb_id, onetile);
                        if (ht == 0) {
                            copy_tile(input_cb_id, 0, k);  // init acc[k] with first row's tile
                        } else {
                            copy_tile(input_cb_id, 0, col_work_dst);
                            detail::sfpu_reduce_max_fold_tile<format>(k, col_work_dst, k);
                        }
                        cb_pop_front(input_cb_id, onetile);
                    }
                }

                sfpu_reduce_init<pool_type, format>();
                for (uint32_t k = 0; k < current_chunk; ++k) {
                    sfpu_reduce<pool_type, format, reduce_dim>(k, /*ct_dim=*/1, /*rt_dim=*/1);
                }

#ifdef REDUCE_POST_MUL
                for (uint32_t k = 0; k < current_chunk; ++k) {
                    detail::sfpu_post_mul_tile<format>(k, post_mul_scaler_bits);
                }
#endif

                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t k = 0; k < current_chunk; ++k) {
                    cb_reserve_back(output_cb_id, onetile);
                    pack_tile(k, output_cb_id);
                    cb_push_back(output_cb_id, onetile);
                }
                tile_regs_release();
            }
        }
    } else {
        // =================================================================
        // REDUCE_ROW: W reduction - each row -> 1 output tile (Ht outputs per batch)
        // =================================================================
        const bool needs_cross_tile_fold = Wt > 1;

        for (uint32_t nc = 0; nc < NC; ++nc) {
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                tile_regs_acquire();

                if (needs_cross_tile_fold) {
                    detail::sfpu_reduce_max_fold_init<format>();
                }

                cb_wait_front(input_cb_id, onetile);
                copy_tile(input_cb_id, 0, dst_idx);
                cb_pop_front(input_cb_id, onetile);

                for (uint32_t wt = 1; wt < Wt; ++wt) {
                    cb_wait_front(input_cb_id, onetile);
                    copy_tile(input_cb_id, 0, next_dst_idx);
                    detail::sfpu_reduce_max_fold_tile<format>(dst_idx, next_dst_idx, dst_idx);
                    cb_pop_front(input_cb_id, onetile);
                }

                sfpu_reduce_init<pool_type, format>();
                sfpu_reduce<pool_type, format, reduce_dim>(dst_idx, /*ct_dim=*/1, /*rt_dim=*/1);

#ifdef REDUCE_POST_MUL
                detail::sfpu_post_mul_tile<format>(dst_idx, post_mul_scaler_bits);
#endif

                tile_regs_commit();

                cb_reserve_back(output_cb_id, onetile);
                tile_regs_wait();
                pack_tile(dst_idx, output_cb_id);
                tile_regs_release();
                cb_push_back(output_cb_id, onetile);
            }
        }
    }

    detail::sfpu_reduce_pack_mask_clear();
}

}  // namespace compute_kernel_lib
