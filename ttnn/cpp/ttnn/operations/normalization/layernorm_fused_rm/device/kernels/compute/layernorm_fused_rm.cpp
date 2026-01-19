// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/compute_kernel_hw_startup.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/rsqrt.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/pack.h"

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t W = get_compile_time_arg_val(2);

    const uint32_t num_tile_rows = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_in_rm = tt::CBIndex::c_0;
    constexpr uint32_t cb_in_tiled = tt::CBIndex::c_1;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
    constexpr uint32_t cb_eps = tt::CBIndex::c_3;
    constexpr uint32_t cb_gamma_tiled = tt::CBIndex::c_6;
    constexpr uint32_t cb_beta_tiled = tt::CBIndex::c_7;
    constexpr uint32_t cb_out_rm = tt::CBIndex::c_16;
    constexpr uint32_t cb_centered = tt::CBIndex::c_24;
    constexpr uint32_t cb_mean = tt::CBIndex::c_25;
    constexpr uint32_t cb_var = tt::CBIndex::c_26;
    constexpr uint32_t cb_invstd = tt::CBIndex::c_27;

    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t onetile = 1;

    compute_kernel_hw_startup(cb_in_rm, cb_in_tiled);

    cb_wait_front(cb_scaler, onetile);
    cb_wait_front(cb_eps, onetile);
    cb_wait_front(cb_gamma_tiled, Wt);
    cb_wait_front(cb_beta_tiled, Wt);

    for (uint32_t row = 0; row < num_tile_rows; row++) {
        // Phase 1: Tilize input
        compute_kernel_lib::tilize(cb_in_rm, Wt, cb_in_tiled, 1, 1, 0, TILE_HEIGHT);

        // Phase 2: Compute mean
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputMode::PERSISTENT>(
                cb_in_tiled, cb_scaler, cb_mean, compute_kernel_lib::TileShape::row(Wt));

        // Phase 3: Center values (x - mean)
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputMode::STREAMING,
            compute_kernel_lib::BinaryDataFormatReconfig::BOTH,
            true,
            true>(
            cb_in_tiled,
            cb_mean,
            cb_centered,
            compute_kernel_lib::BinaryTileShape::row(Wt),
            compute_kernel_lib::BinaryTileLayout::contiguous());

        // Phase 4: Square centered values
        compute_kernel_lib::square<
            compute_kernel_lib::BinaryInputMode::STREAMING,
            compute_kernel_lib::BinaryDataFormatReconfig::BOTH,
            true,
            true>(
            cb_centered,
            cb_in_tiled,
            compute_kernel_lib::BinaryTileShape::row(Wt),
            compute_kernel_lib::BinaryTileLayout::contiguous());

        // Phase 5: Compute variance
        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
            cb_in_tiled, cb_scaler, cb_var, compute_kernel_lib::TileShape::row(Wt));

        // Phase 6: Add epsilon and compute rsqrt
        compute_kernel_lib::add<compute_kernel_lib::BroadcastDim::SCALAR>(
            cb_var, cb_eps, cb_invstd, compute_kernel_lib::BinaryTileShape::single());

        {
            rsqrt_tile_init();
            tile_regs_acquire();
            cb_wait_front(cb_invstd, onetile);
            copy_tile_to_dst_init_short(cb_invstd);
            copy_tile(cb_invstd, 0, 0);
            rsqrt_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_var, onetile);
            pack_tile(0, cb_var);
            cb_push_back(cb_var, onetile);
            tile_regs_release();
            cb_pop_front(cb_invstd, onetile);
        }

        // Phase 7: Re-tilize and re-compute centered values
        compute_kernel_lib::tilize(cb_in_rm, Wt, cb_in_tiled, 1, 1, 0, TILE_HEIGHT);

        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputMode::PERSISTENT>(
                cb_in_tiled, cb_scaler, cb_mean, compute_kernel_lib::TileShape::row(Wt));

        compute_kernel_lib::sub<compute_kernel_lib::BroadcastDim::COL>(
            cb_in_tiled, cb_mean, cb_centered, compute_kernel_lib::BinaryTileShape::row(Wt));

        // Phase 8: Normalize (centered * inv_std)
        compute_kernel_lib::mul<compute_kernel_lib::BroadcastDim::COL>(
            cb_centered, cb_var, cb_in_tiled, compute_kernel_lib::BinaryTileShape::row(Wt));

        // Phase 9: Apply gamma
        compute_kernel_lib::mul<compute_kernel_lib::BroadcastDim::ROW>(
            cb_in_tiled, cb_gamma_tiled, cb_centered, compute_kernel_lib::BinaryTileShape::row(Wt));

        // Phase 10: Apply beta
        compute_kernel_lib::add<compute_kernel_lib::BroadcastDim::ROW>(
            cb_centered, cb_beta_tiled, cb_in_tiled, compute_kernel_lib::BinaryTileShape::row(Wt));

        // Phase 11: Untilize output
        compute_kernel_lib::untilize<Wt, cb_in_tiled, cb_out_rm>(1);
    }
}
}  // namespace NAMESPACE
