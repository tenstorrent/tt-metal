// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm compute. Per block of block_rows tile-rows (whole row resident):
//   P1 mean_block   : mean[r] = rowsum(x) / W                      -> cb_row_stat
//   P2 center_block : xc = x - mean (col bcast)                     -> cb_centered
//   P3 sqsum_block  : S[r] = sum_t xc[r,t]^2 (packer L1 accumulate)   -> cb_row_stat
//   P4 rstd_block   : rstd[r] = rsqrt(rowsum(S[r]) / W + eps)        -> cb_rstd
//   P5 scale_block  : xc *= rstd (col bcast, in place)
//   P6 gamma_block  : xc *= gamma (row bcast, in place)
//   P7 shift_block  : out = xc + beta (row bcast)                   -> cb_output_tiles
//
// The P4 finalize post-op (mul_unary / add_unary / rsqrt) uses raw SFPU calls because reduce()'s
// post_reduce_op slot takes a DEST-index callable, not chain elements.
// The reduce scaler slot names cb_gamma_rows: AccumulateViaAdd on an aligned reduce never reads it.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gamma_rows = get_compile_time_arg_val(1);
    constexpr uint32_t cb_beta_rows = get_compile_time_arg_val(2);
    constexpr uint32_t cb_row_stat = get_compile_time_arg_val(3);
    constexpr uint32_t cb_rstd = get_compile_time_arg_val(4);
    constexpr uint32_t cb_centered = get_compile_time_arg_val(5);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t block_rows = get_compile_time_arg_val(7);
    constexpr uint32_t block_w_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t width = get_compile_time_arg_val(9);
    constexpr uint32_t inv_w_bits = get_compile_time_arg_val(10);

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);
    const uint32_t eps_bits = get_arg_val<uint32_t>(1);

    using namespace compute_kernel_lib;

    compute_kernel_hw_startup(cb_input_tiles, cb_gamma_rows, cb_row_stat);

    DataflowBuffer input_tiles(cb_input_tiles);
    DataflowBuffer gamma_rows(cb_gamma_rows);
    DataflowBuffer beta_rows(cb_beta_rows);
    DataflowBuffer row_stat(cb_row_stat);
    DataflowBuffer rstd(cb_rstd);
    DataflowBuffer centered(cb_centered);
    DataflowBuffer output_tiles(cb_output_tiles);

    constexpr auto block_shape = IterationShape::grid(block_rows, block_w_tiles);

    constexpr InputSpec centered_row = input(
        cb_centered,
        WaitPolicy::Upfront,
        PopPolicy::None,
        InputTileMapping::Block,
        DataFormatReconfig::Enabled,
        TileAddressing::Offset);
    using SquareRow = BinaryFpu<BinaryFpuOp::Mul, centered_row, centered_row>;
    using SumsqPack = PackTile<output(
        cb_row_stat,
        ReservePolicy::OneUpfront,
        PushPolicy::OneAtEnd,
        DataFormatReconfig::Enabled,
        TileAddressing::Direct,
        DestAccumulation::Disabled,
        L1Accumulation::Enabled)>;

    auto finalize_rstd = [eps_bits](uint32_t dst_idx) {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst_idx, inv_w_bits);
        add_unary_tile(dst_idx, eps_bits);
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    };

    for (uint32_t block = 0; block < num_blocks; ++block) {
        // P1 mean_block: the input block stays resident for P2.
        reduce_mean<
            ReduceDim::REDUCE_ROW,
            cb_input_tiles,
            cb_gamma_rows,
            cb_row_stat,
            ReduceInputPolicy::WaitUpfrontNoPop,
            ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
            ReduceFp32Mode::Fast,
            ReduceAlgorithm::AccumulateViaAdd>(ReduceInputBlockShape::of(block_rows, block_w_tiles), width);

        // P2 center_block
        sub<input(cb_input_tiles),
            input(cb_row_stat, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
            output(cb_centered)>(block_shape, input_tiles, row_stat, centered);

        // P3 sqsum_block: one L1-accumulated tile per tile-row; cb_centered stays resident.
        for (uint32_t row = 0; row < block_rows; ++row) {
            const uint32_t row_base = row * block_w_tiles;
            eltwise_chain(
                IterationShape::tiles(block_w_tiles),
                SquareRow{centered, centered, row_base, row_base},
                SumsqPack{row_stat});
        }

        // P4 rstd_block
        reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            cb_row_stat,
            cb_gamma_rows,
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

        // P5 scale_block (in place)
        mul<input(cb_centered),
            input(cb_rstd, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
            output(cb_centered)>(block_shape, centered, rstd, centered);

        // P6 gamma_block (in place)
        mul<input(cb_centered),
            input(cb_gamma_rows, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row),
            output(cb_centered)>(block_shape, centered, gamma_rows, centered);

        // P7 shift_block
        add<input(cb_centered),
            input(cb_beta_rows, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row),
            output(cb_output_tiles)>(block_shape, centered, beta_rows, output_tiles);
    }

    gamma_rows.pop_front(block_w_tiles);
    beta_rows.pop_front(block_w_tiles);
}
