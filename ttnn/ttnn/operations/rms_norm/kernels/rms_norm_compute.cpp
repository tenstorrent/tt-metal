// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm compute. Per block of R tile-rows (whole row resident in cb_input_tiles):
//   row_sumsq_block : S[r] = sum_t x[r,t]^2, accumulated in DEST (masked last column summed separately)
//   row_rinv_block  : rinv[r] = rsqrt(rowsum(S[r]) / W + eps)
//   normalize_block : per W-chunk, y = x * rinv (col bcast) [* gamma (row bcast)]
//
// The finalize post-op (mul_unary / add_unary / rsqrt) uses raw SFPU calls because reduce()'s
// post_reduce_op slot takes a DEST-index callable, not chain elements (see op_design.md, API Mapping).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gamma_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sumsq = get_compile_time_arg_val(3);
    constexpr uint32_t cb_rinv = get_compile_time_arg_val(4);
    constexpr uint32_t cb_x_normed = get_compile_time_arg_val(5);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t block_row_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t w_chunk_tiles = get_compile_time_arg_val(9);
    constexpr bool has_partial_w = get_compile_time_arg_val(10) != 0;
    constexpr bool has_gamma = get_compile_time_arg_val(11) != 0;
    constexpr uint32_t inv_w_bits = get_compile_time_arg_val(12);
    constexpr uint32_t epsilon_bits = get_compile_time_arg_val(13);

    const uint32_t core_rows = get_arg_val<uint32_t>(0);

    using namespace compute_kernel_lib;

    constexpr uint32_t aligned_w_tiles = tensor_w_tiles - (has_partial_w ? 1 : 0);
    constexpr uint32_t block_tiles = block_row_tiles * tensor_w_tiles;
    constexpr uint32_t normalize_dst_cb = has_gamma ? cb_x_normed : cb_output_tiles;

    compute_kernel_hw_startup(cb_input_tiles, cb_input_tiles, cb_sumsq);

    DataflowBuffer input_tiles(cb_input_tiles);
    DataflowBuffer sumsq(cb_sumsq);
    DataflowBuffer rinv(cb_rinv);
    DataflowBuffer x_normed(cb_x_normed);
    DataflowBuffer gamma_tiles(cb_gamma_tiles);
    DataflowBuffer output_tiles(cb_output_tiles);
    DataflowBuffer scaler(cb_scaler);

    constexpr InputSpec x_strided = input(
        cb_input_tiles,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Block,
        DataFormatReconfig::Enabled,
        TileAddressing::Strided);
    using SquareAccumulate = BinaryFpu<BinaryFpuOp::Mul, x_strided, x_strided, Dst::D0, DestAccumulation::PerRow>;
    using SumsqPack = PackTile<output(
        cb_sumsq,
        ReservePolicy::PerOuter,
        PushPolicy::PerOuter,
        DataFormatReconfig::Enabled,
        TileAddressing::Direct,
        DestAccumulation::PerRow)>;
    using MulRinv = BinaryFpu<
        BinaryFpuOp::Mul,
        x_strided,
        input(cb_rinv, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Col)>;
    using NormalizePack = PackTile<output(normalize_dst_cb)>;

    auto finalize_rinv = [](uint32_t dst_idx) {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst_idx, inv_w_bits);
        add_unary_tile(dst_idx, epsilon_bits);
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    };

    for (uint32_t block_row = 0; block_row < core_rows; block_row += block_row_tiles) {
        const uint32_t rows = (core_rows - block_row) < block_row_tiles ? (core_rows - block_row) : block_row_tiles;

        input_tiles.wait_front(block_tiles);

        // row_sumsq_block
        if constexpr (aligned_w_tiles > 0) {
            eltwise_chain(
                IterationShape::grid(rows, aligned_w_tiles),
                SquareAccumulate{
                    input_tiles, input_tiles, StridedTileRange{0, tensor_w_tiles}, StridedTileRange{0, tensor_w_tiles}},
                SumsqPack{sumsq});
        }
        if constexpr (has_partial_w) {
            eltwise_chain(
                IterationShape::grid(rows, 1),
                SquareAccumulate{
                    input_tiles,
                    input_tiles,
                    StridedTileRange{tensor_w_tiles - 1, tensor_w_tiles},
                    StridedTileRange{tensor_w_tiles - 1, tensor_w_tiles}},
                SumsqPack{sumsq});
        }

        // row_rinv_block
        const auto rinv_shape = ReduceInputBlockShape::of(rows, 1);
        if constexpr (has_partial_w && aligned_w_tiles > 0) {
            reduce<
                PoolType::SUM,
                ReduceDim::REDUCE_ROW,
                cb_sumsq,
                cb_scaler,
                cb_rinv,
                ReduceInputPolicy::BulkWaitBulkPop,
                ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                ReduceFp32Mode::Fast,
                ReduceAlgorithm::ReduceTile>(rinv_shape, ReduceInputMemoryLayout::contiguous());
            reduce<
                PoolType::SUM,
                ReduceDim::REDUCE_ROW,
                cb_sumsq,
                cb_scaler,
                cb_rinv,
                ReduceInputPolicy::BulkWaitBulkPop,
                ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                ReduceFp32Mode::Fast,
                ReduceAlgorithm::ReduceTile>(
                rinv_shape,
                ReduceInputMemoryLayout::contiguous(),
                Accumulate::at(cb_rinv, 1),
                finalize_rinv,
                ReducePartialScaler::with_partial());
        } else {
            reduce<
                PoolType::SUM,
                ReduceDim::REDUCE_ROW,
                cb_sumsq,
                cb_scaler,
                cb_rinv,
                ReduceInputPolicy::BulkWaitBulkPop,
                ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                ReduceFp32Mode::Fast,
                ReduceAlgorithm::ReduceTile>(
                rinv_shape,
                ReduceInputMemoryLayout::contiguous(),
                NoAccumulation{},
                finalize_rinv,
                has_partial_w ? ReducePartialScaler::with_partial() : ReducePartialScaler::none());
        }

        // normalize_block
        for (uint32_t chunk_start = 0; chunk_start < tensor_w_tiles; chunk_start += w_chunk_tiles) {
            const uint32_t cols =
                (tensor_w_tiles - chunk_start) < w_chunk_tiles ? (tensor_w_tiles - chunk_start) : w_chunk_tiles;
            const auto chunk_shape = IterationShape::grid(rows, cols);

            if constexpr (has_gamma) {
                eltwise_chain(
                    chunk_shape,
                    MulRinv{input_tiles, rinv, StridedTileRange{chunk_start, tensor_w_tiles}},
                    NormalizePack{x_normed});
                gamma_tiles.wait_front(w_chunk_tiles);
                mul<input(cb_x_normed),
                    input(cb_gamma_tiles, BroadcastDim::Row, WaitPolicy::None, PopPolicy::None, InputTileMapping::Row),
                    output(cb_output_tiles)>(chunk_shape, x_normed, gamma_tiles, output_tiles);
                gamma_tiles.pop_front(w_chunk_tiles);
            } else {
                eltwise_chain(
                    chunk_shape,
                    MulRinv{input_tiles, rinv, StridedTileRange{chunk_start, tensor_w_tiles}},
                    NormalizePack{output_tiles});
            }
        }

        input_tiles.pop_front(block_tiles);
        rinv.pop_front(rows);
    }

    scaler.pop_front(has_partial_w ? 2 : 1);
}
