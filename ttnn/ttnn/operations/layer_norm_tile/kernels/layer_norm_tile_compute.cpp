// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_tile compute. Per block of block_rows tile-rows (whole W resident):
//   add_residual_block : xsum = x + residual                         (HAS_RESIDUAL only)
//   mean_block         : mean = rowsum(x') / W                       (x' stays resident)
//   center_block       : c = x' - mean                (Col bcast)
//   sumsq_block        : S = sum_t c[t]^2              (DEST-accumulated per tile-row)
//   rstd_block         : rstd = rsqrt(rowsum(S) / W + eps)
//   normalize_block    : n = c * rstd                 (Col bcast)
//   scale_block        : s = n * gamma                (Row bcast)
//   shift_block        : y = s + beta                 (Row bcast)
//
// cb_xsum / cb_centered / cb_normed / cb_scaled share one allocation (alias group `work`), as do
// cb_mean / cb_sumsq / cb_rstd (`stats`); every id of a group moves the same page count per block.
//
// rstd_block's post-op calls the SFPU scalar ops directly: reduce()'s post_reduce_op is a DEST-index
// callable, not a chain (op_design.md, API Mapping).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_residual_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gamma_rows = get_compile_time_arg_val(2);
    constexpr uint32_t cb_beta_rows = get_compile_time_arg_val(3);
    constexpr uint32_t cb_xsum = get_compile_time_arg_val(4);
    constexpr uint32_t cb_centered = get_compile_time_arg_val(5);
    constexpr uint32_t cb_normed = get_compile_time_arg_val(6);
    constexpr uint32_t cb_scaled = get_compile_time_arg_val(7);
    constexpr uint32_t cb_mean = get_compile_time_arg_val(8);
    constexpr uint32_t cb_sumsq = get_compile_time_arg_val(9);
    constexpr uint32_t cb_rstd = get_compile_time_arg_val(10);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(11);
    constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(12);
    constexpr uint32_t block_height = get_compile_time_arg_val(13);
    constexpr bool has_residual = get_compile_time_arg_val(14) != 0;
    constexpr uint32_t width = get_compile_time_arg_val(15);
    constexpr uint32_t inv_w_bits = get_compile_time_arg_val(16);

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);
    const uint32_t last_block_extent = get_arg_val<uint32_t>(1);
    const uint32_t eps_bits = get_arg_val<uint32_t>(2);

    using namespace compute_kernel_lib;

    // The aligned AccumulateViaAdd reduce never reads its scaler CB; it only needs a configured id.
    constexpr uint32_t cb_unused_scaler = cb_gamma_rows;
    constexpr uint32_t cb_x = has_residual ? cb_xsum : cb_input_tiles;

    compute_kernel_hw_startup(cb_input_tiles, cb_input_tiles, cb_centered);

    DataflowBuffer x_tiles(cb_x);
    DataflowBuffer gamma_rows(cb_gamma_rows);
    DataflowBuffer beta_rows(cb_beta_rows);
    DataflowBuffer centered(cb_centered);
    DataflowBuffer normed(cb_normed);
    DataflowBuffer scaled(cb_scaled);
    DataflowBuffer mean(cb_mean);
    DataflowBuffer sumsq(cb_sumsq);
    DataflowBuffer output_tiles(cb_output_tiles);

    using SquareAccumulate = BinaryFpu<
        BinaryFpuOp::Mul,
        input(cb_centered, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Block),
        input(cb_centered, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Block),
        Dst::D0,
        DestAccumulation::PerRow>;
    using SumsqPack = PackTile<output(
        cb_sumsq,
        ReservePolicy::PerOuter,
        PushPolicy::PerOuter,
        DataFormatReconfig::Enabled,
        TileAddressing::Direct,
        DestAccumulation::PerRow)>;

    auto finalize_rstd = [eps_bits](uint32_t dst_idx) {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst_idx, inv_w_bits);
        add_unary_tile(dst_idx, eps_bits);
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    };

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t block_rows = (block_idx + 1 == num_blocks) ? last_block_extent : block_height;
        const auto block_grid = IterationShape::grid(block_rows, tensor_w_tiles);

        // add_residual_block
        if constexpr (has_residual) {
            DataflowBuffer input_tiles(cb_input_tiles);
            DataflowBuffer residual_tiles(cb_residual_tiles);
            add<input(cb_input_tiles), input(cb_residual_tiles), output(cb_xsum)>(
                IterationShape::tiles(block_rows * tensor_w_tiles), input_tiles, residual_tiles, x_tiles);
        }

        // mean_block: x' stays resident for center_block.
        reduce_mean<ReduceDim::REDUCE_ROW, cb_x, cb_unused_scaler, cb_mean, ReduceInputPolicy::WaitUpfrontNoPop>(
            ReduceInputBlockShape::of(block_rows, tensor_w_tiles), width);

        // center_block
        sub<input(cb_x),
            input(cb_mean, BroadcastDim::Col, WaitPolicy::PerTile, PopPolicy::PerTile, InputTileMapping::Col),
            output(cb_centered)>(block_grid, x_tiles, mean, centered);

        // sumsq_block: centered stays resident for normalize_block.
        eltwise_chain(block_grid, SquareAccumulate{centered, centered}, SumsqPack{sumsq});

        // rstd_block
        reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            cb_sumsq,
            cb_unused_scaler,
            cb_rstd,
            ReduceInputPolicy::BulkWaitBulkPop,
            ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
            ReduceFp32Mode::Fast,
            ReduceAlgorithm::AccumulateViaAdd,
            ReduceWithinTile::Collapse>(
            ReduceInputBlockShape::of(block_rows, 1),
            ReduceInputMemoryLayout::contiguous(),
            NoAccumulation{},
            finalize_rstd);

        // normalize_block
        {
            DataflowBuffer rstd(cb_rstd);
            mul<input(cb_centered),
                input(cb_rstd, BroadcastDim::Col, WaitPolicy::PerTile, PopPolicy::PerTile, InputTileMapping::Col),
                output(cb_normed)>(block_grid, centered, rstd, normed);
        }

        // scale_block
        mul<input(cb_normed),
            input(cb_gamma_rows, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row),
            output(cb_scaled)>(block_grid, normed, gamma_rows, scaled);

        // shift_block
        add<input(cb_scaled),
            input(cb_beta_rows, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row),
            output(cb_output_tiles)>(block_grid, scaled, beta_rows, output_tiles);
    }

    gamma_rows.pop_front(tensor_w_tiles);
    beta_rows.pop_front(tensor_w_tiles);
}
