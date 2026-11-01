// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm compute. Per block of R_b tile-rows (whole row resident):
//   residual_add_block : z = x + res                         -> work_a   (HAS_RESIDUAL)
//   row_mean_block     : mean = sum_w z / W                  -> row_mean (z stays resident)
//   center_block       : c = z - mean (col bcast)            -> work_b
//   row_sumsq_block    : S = sum_w c * c, DEST-accumulated   -> row_sumsq (c stays resident)
//   row_inv_std_block  : rsqrt(collapse(S) / W + eps)        -> inv_std
//   normalize_block    : c * inv_std (col bcast)             -> work_a
//   affine_scale_block : * gamma (row bcast)                 -> work_b
//   affine_shift_block : + beta (row bcast)                  -> output
//
// work_a / work_b alias one allocation; every phase moves whole blocks through each id so page i of
// one is page i of the other. Both reduces use AccumulateViaAdd on tile-aligned W, which never reads
// the scaler slot (bound to cb_gamma_tiles). The inv_std post-op uses raw SFPU calls because
// reduce()'s post_reduce_op hook takes a DEST-index callable, not chain elements.

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
    constexpr uint32_t cb_gamma_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t cb_beta_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t cb_row_work_a = get_compile_time_arg_val(4);
    constexpr uint32_t cb_row_work_b = get_compile_time_arg_val(5);
    constexpr uint32_t cb_row_mean = get_compile_time_arg_val(6);
    constexpr uint32_t cb_row_sumsq = get_compile_time_arg_val(7);
    constexpr uint32_t cb_inv_std = get_compile_time_arg_val(8);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t block_row_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(11);
    constexpr bool has_residual = get_compile_time_arg_val(12) != 0;
    constexpr uint32_t width = get_compile_time_arg_val(13);
    constexpr uint32_t inv_w_bits = get_compile_time_arg_val(14);
    constexpr uint32_t epsilon_bits = get_compile_time_arg_val(15);

    const uint32_t core_row_tiles = get_arg_val<uint32_t>(0);

    using namespace compute_kernel_lib;

    // Source of the mean and the centering: z = x + res when present, else x itself.
    constexpr uint32_t cb_z = has_residual ? cb_row_work_a : cb_input_tiles;

    compute_kernel_hw_startup(cb_input_tiles, cb_gamma_tiles, cb_output_tiles);

    DataflowBuffer input_tiles(cb_input_tiles);
    [[maybe_unused]] DataflowBuffer residual_tiles(cb_residual_tiles);
    DataflowBuffer gamma_tiles(cb_gamma_tiles);
    DataflowBuffer beta_tiles(cb_beta_tiles);
    DataflowBuffer work_a(cb_row_work_a);
    DataflowBuffer work_b(cb_row_work_b);
    DataflowBuffer row_mean(cb_row_mean);
    DataflowBuffer row_sumsq(cb_row_sumsq);
    DataflowBuffer inv_std(cb_inv_std);
    DataflowBuffer output_tiles(cb_output_tiles);
    DataflowBuffer& z_tiles = has_residual ? work_a : input_tiles;

    constexpr InputSpec centered_resident =
        input(cb_row_work_b, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block);
    using SquareAccumulate =
        BinaryFpu<BinaryFpuOp::Mul, centered_resident, centered_resident, Dst::D0, DestAccumulation::PerRow>;
    using SumsqPack = PackTile<output(
        cb_row_sumsq,
        ReservePolicy::PerOuter,
        PushPolicy::PerOuter,
        DataFormatReconfig::Enabled,
        TileAddressing::Direct,
        DestAccumulation::PerRow)>;

    auto finalize_inv_std = [](uint32_t dst_idx) {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst_idx, inv_w_bits);
        add_unary_tile(dst_idx, epsilon_bits);
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    };

    for (uint32_t block_row = 0; block_row < core_row_tiles; block_row += block_row_tiles) {
        const uint32_t rows =
            (core_row_tiles - block_row) < block_row_tiles ? (core_row_tiles - block_row) : block_row_tiles;
        const auto block_shape = IterationShape::grid(rows, tensor_w_tiles);

        // residual_add_block
        if constexpr (has_residual) {
            add<input(cb_input_tiles), input(cb_residual_tiles), output(cb_row_work_a)>(
                block_shape, input_tiles, residual_tiles, work_a);
        }

        // row_mean_block (source left resident for center_block)
        reduce_mean<
            ReduceDim::REDUCE_ROW,
            cb_z,
            cb_gamma_tiles,
            cb_row_mean,
            ReduceInputPolicy::WaitUpfrontNoPop,
            ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
            ReduceFp32Mode::Fast,
            ReduceAlgorithm::AccumulateViaAdd>(ReduceInputBlockShape::of(rows, tensor_w_tiles), width);

        // center_block
        sub<input(cb_z),
            input(cb_row_mean, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
            output(cb_row_work_b)>(block_shape, z_tiles, row_mean, work_b);

        // row_sumsq_block (c stays resident for normalize_block)
        work_b.wait_front(rows * tensor_w_tiles);
        eltwise_chain(block_shape, SquareAccumulate{work_b, work_b}, SumsqPack{row_sumsq});

        // row_inv_std_block
        reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            cb_row_sumsq,
            cb_gamma_tiles,
            cb_inv_std,
            ReduceInputPolicy::BulkWaitBulkPop,
            ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
            ReduceFp32Mode::Fast,
            ReduceAlgorithm::AccumulateViaAdd,
            ReduceWithinTile::Collapse>(
            ReduceInputBlockShape::of(rows, 1),
            ReduceInputMemoryLayout::contiguous(),
            NoAccumulation{},
            finalize_inv_std);

        // normalize_block
        mul<input(cb_row_work_b),
            input(cb_inv_std, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
            output(cb_row_work_a)>(block_shape, work_b, inv_std, work_a);

        // affine_scale_block
        mul<input(cb_row_work_a),
            input(cb_gamma_tiles, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row),
            output(cb_row_work_b)>(block_shape, work_a, gamma_tiles, work_b);

        // affine_shift_block
        add<input(cb_row_work_b),
            input(cb_beta_tiles, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row),
            output(cb_output_tiles)>(block_shape, work_b, beta_tiles, output_tiles);
    }

    gamma_tiles.pop_front(tensor_w_tiles);
    beta_tiles.pop_front(tensor_w_tiles);
}
