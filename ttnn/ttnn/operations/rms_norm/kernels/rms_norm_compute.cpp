// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm compute (TRISC0/1/2).
//
// Per block (block_rows tile-rows x this core's resident W slice), op_design.md §Block schedule:
//   [RM] tilize          cb_x_rm_sticks -> cb_x_tiles            (one tilize call per block)
//   sumsq_block          cb_x_tiles (resident, not popped) -> cb_sumsq   per row [P_full = sum x^2 over all but
//                        the last valid tile, P_last = x^2 of the last valid tile]; x*x accumulates in DEST (PerRow)
//   rowsum_block         cb_sumsq -> R1: cb_inv_rms = rsqrt(S/W + eps);  R2: cb_rowsum_partial = S
//                        (ReduceTile, partial scaler on P_last masks the padding lanes: cap M8)
//   scale_gamma_block    first xg_lookahead_chunks chunks: cb_x_tiles (x) * cb_gamma_tiles (row bcast) -> cb_x_gamma
//   [R2 root] combine    cb_rowsum_gather (block_rows x w_group_size partials) -> cb_inv_rms_root
//                        (AccumulateViaAdd + ReduceWithinTile::Skip, same finalize post-op)
//   normalize_block      per chunk: (cb_x_gamma | cb_x_tiles) * cb_inv_rms (col bcast) -> cb_out_tiles,
//                        then the next x*gamma chunk beyond the lookahead
//   [RM] untilize        per chunk: cb_out_tiles -> cb_out_rm_sticks
//
// Every phase is a kernel-lib helper (eltwise_chain / mul / reduce / tilize / untilize); no raw LLK.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_x_rm_sticks = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gamma_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t cb_reduce_scaler = get_compile_time_arg_val(3);
    constexpr uint32_t cb_sumsq = get_compile_time_arg_val(4);
    constexpr uint32_t cb_rowsum_partial = get_compile_time_arg_val(5);
    constexpr uint32_t cb_rowsum_gather = get_compile_time_arg_val(6);
    constexpr uint32_t cb_inv_rms_root = get_compile_time_arg_val(7);
    constexpr uint32_t cb_inv_rms = get_compile_time_arg_val(8);
    constexpr uint32_t cb_x_gamma = get_compile_time_arg_val(9);
    constexpr uint32_t cb_out_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t cb_out_rm_sticks = get_compile_time_arg_val(11);
    constexpr bool is_rm = get_compile_time_arg_val(12) != 0;
    constexpr bool has_gamma = get_compile_time_arg_val(13) != 0;
    constexpr bool is_w_split = get_compile_time_arg_val(14) != 0;  // regime R2
    constexpr uint32_t block_rows = get_compile_time_arg_val(15);
    constexpr uint32_t chunk_w_tiles = get_compile_time_arg_val(16);
    constexpr uint32_t chunks_per_row = get_compile_time_arg_val(17);
    constexpr uint32_t xg_lookahead_chunks = get_compile_time_arg_val(18);
    constexpr uint32_t w_group_size = get_compile_time_arg_val(19);

    constexpr uint32_t row_stride_tiles = chunks_per_row * chunk_w_tiles;

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);
    const uint32_t slice_w_tiles = get_arg_val<uint32_t>(1);
    const bool is_root = get_arg_val<uint32_t>(2) != 0;
    const uint32_t inv_w_bits = get_arg_val<uint32_t>(3);
    const uint32_t eps_bits = get_arg_val<uint32_t>(4);

    using namespace ckl;

    if constexpr (is_rm) {
        compute_kernel_hw_startup(cb_x_rm_sticks, cb_x_rm_sticks, cb_x_tiles);
    } else {
        compute_kernel_hw_startup(cb_x_tiles, cb_x_tiles, cb_sumsq);
    }

    // mean -> +eps -> rsqrt, fused as the reduce post-op on the row-sum in DEST.
    auto finalize_inv_rms = [inv_w_bits, eps_bits](uint32_t dst) {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst, inv_w_bits);
        add_unary_tile(dst, eps_bits);
        rsqrt_tile_init();
        rsqrt_tile(dst);
    };

    // [P_full, P_last] per row; P_full is absent when the slice holds a single tile.
    const uint32_t cols_per_row = slice_w_tiles > 1 ? 2 : 1;
    constexpr auto partial = ReducePartialScaler::with_partial();

    if constexpr (has_gamma) {
        cb_wait_front(cb_gamma_tiles, row_stride_tiles);  // resident for the whole kernel
    }

    // scale_gamma_block for one chunk: x chunk is the FIFO front (earlier chunks already popped; M5),
    // gamma is addressed at the chunk's column offset.
    auto scale_gamma_chunk = [](uint32_t chunk_idx) {
        if constexpr (has_gamma) {
            eltwise_chain(
                IterationShape::grid(block_rows, chunk_w_tiles),
                BinaryFpu<
                    BinaryFpuOp::Mul,
                    input(cb_x_tiles, WaitPolicy::None, PopPolicy::AtEnd, InputTileMapping::Block),
                    input(
                        cb_gamma_tiles,
                        BroadcastDim::Row,
                        WaitPolicy::None,
                        PopPolicy::None,
                        InputTileMapping::Row,
                        DataFormatReconfig::Enabled,
                        TileAddressing::Offset)>{0u, chunk_idx * chunk_w_tiles},
                PackTile<output(cb_x_gamma)>{});
        }
    };

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        // ---- load_block (RM part): one tilize call per block ----
        if constexpr (is_rm) {
            tilize<chunk_w_tiles, cb_x_rm_sticks, cb_x_tiles>(block_rows * chunks_per_row);
        }
        cb_wait_front(cb_x_tiles, block_rows * row_stride_tiles);

        // ---- sumsq_block: x*x accumulated across the slice in DEST, per row ----
        cb_reserve_back(cb_sumsq, block_rows * cols_per_row);
        if (slice_w_tiles > 1) {
            eltwise_chain(
                IterationShape::grid(block_rows, slice_w_tiles - 1),
                BinaryFpu<
                    BinaryFpuOp::Mul,
                    input(
                        cb_x_tiles,
                        WaitPolicy::None,
                        PopPolicy::None,
                        InputTileMapping::Block,
                        DataFormatReconfig::Enabled,
                        TileAddressing::Strided),
                    input(
                        cb_x_tiles,
                        WaitPolicy::None,
                        PopPolicy::None,
                        InputTileMapping::Block,
                        DataFormatReconfig::Enabled,
                        TileAddressing::Strided),
                    Dst::D0,
                    DestAccumulation::PerRow>{
                    StridedTileRange{0, row_stride_tiles}, StridedTileRange{0, row_stride_tiles}},
                PackTile<output(
                    cb_sumsq,
                    ReservePolicy::None,
                    PushPolicy::None,
                    DataFormatReconfig::Enabled,
                    TileAddressing::Strided,
                    DestAccumulation::PerRow)>{StridedTileRange{0, cols_per_row}});
        }
        eltwise_chain(
            IterationShape::grid(block_rows, 1),
            BinaryFpu<
                BinaryFpuOp::Mul,
                input(
                    cb_x_tiles,
                    WaitPolicy::None,
                    PopPolicy::None,
                    InputTileMapping::Block,
                    DataFormatReconfig::Enabled,
                    TileAddressing::Strided),
                input(
                    cb_x_tiles,
                    WaitPolicy::None,
                    PopPolicy::None,
                    InputTileMapping::Block,
                    DataFormatReconfig::Enabled,
                    TileAddressing::Strided),
                Dst::D0,
                DestAccumulation::PerRow>{
                StridedTileRange{slice_w_tiles - 1, row_stride_tiles},
                StridedTileRange{slice_w_tiles - 1, row_stride_tiles}},
            PackTile<output(
                cb_sumsq,
                ReservePolicy::None,
                PushPolicy::None,
                DataFormatReconfig::Enabled,
                TileAddressing::Strided,
                DestAccumulation::PerRow)>{StridedTileRange{cols_per_row - 1, cols_per_row}});
        cb_push_back(cb_sumsq, block_rows * cols_per_row);

        // ---- rowsum_block ----
        if constexpr (is_w_split) {
            reduce<
                PoolType::SUM,
                ReduceDim::REDUCE_ROW,
                cb_sumsq,
                cb_reduce_scaler,
                cb_rowsum_partial,
                ReduceInputPolicy::BulkWaitBulkPop,
                ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                ReduceFp32Mode::Fast,
                ReduceAlgorithm::ReduceTile>(
                ReduceInputBlockShape::of(block_rows, cols_per_row),
                ReduceInputMemoryLayout::contiguous(),
                NoAccumulation{},
                NoOp{},
                partial);
        } else {
            reduce<
                PoolType::SUM,
                ReduceDim::REDUCE_ROW,
                cb_sumsq,
                cb_reduce_scaler,
                cb_inv_rms,
                ReduceInputPolicy::BulkWaitBulkPop,
                ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                ReduceFp32Mode::Fast,
                ReduceAlgorithm::ReduceTile>(
                ReduceInputBlockShape::of(block_rows, cols_per_row),
                ReduceInputMemoryLayout::contiguous(),
                NoAccumulation{},
                finalize_inv_rms,
                partial);
        }

        // ---- scale_gamma_block (lookahead): fills the combine stall in R2 (S1 / S1') ----
        if constexpr (has_gamma) {
            for (uint32_t ci = 0; ci < xg_lookahead_chunks; ++ci) {
                scale_gamma_chunk(ci);
            }
        }

        // ---- combine_block (R2 root): sum the w_group_size partials of each row, finalize ----
        if constexpr (is_w_split) {
            if (is_root) {
                reduce<
                    PoolType::SUM,
                    ReduceDim::REDUCE_ROW,
                    cb_rowsum_gather,
                    cb_reduce_scaler,
                    cb_inv_rms_root,
                    ReduceInputPolicy::BulkWaitBulkPop,
                    ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                    ReduceFp32Mode::Fast,
                    ReduceAlgorithm::AccumulateViaAdd,
                    ReduceWithinTile::Skip>(
                    ReduceInputBlockShape::of(block_rows, w_group_size),
                    ReduceInputMemoryLayout::contiguous(),
                    NoAccumulation{},
                    finalize_inv_rms);
            }
        }

        // ---- normalize_block (+ remaining x*gamma chunks, + RM untilize) ----
        cb_wait_front(cb_inv_rms, block_rows);
        for (uint32_t ci = 0; ci < chunks_per_row; ++ci) {
            if constexpr (has_gamma) {
                mul<input(cb_x_gamma, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block),
                    input(cb_inv_rms, BroadcastDim::Col, WaitPolicy::None, PopPolicy::None, InputTileMapping::Col),
                    output(cb_out_tiles)>(IterationShape::grid(block_rows, chunk_w_tiles));
                if (ci + xg_lookahead_chunks < chunks_per_row) {
                    scale_gamma_chunk(ci + xg_lookahead_chunks);
                }
            } else {
                mul<input(cb_x_tiles, WaitPolicy::None, PopPolicy::AtEnd, InputTileMapping::Block),
                    input(cb_inv_rms, BroadcastDim::Col, WaitPolicy::None, PopPolicy::None, InputTileMapping::Col),
                    output(cb_out_tiles)>(IterationShape::grid(block_rows, chunk_w_tiles));
            }
            if constexpr (is_rm) {
                untilize<chunk_w_tiles, cb_out_tiles, cb_out_rm_sticks>(block_rows);
            }
        }
        cb_pop_front(cb_inv_rms, block_rows);
    }

    if constexpr (has_gamma) {
        cb_pop_front(cb_gamma_tiles, row_stride_tiles);
    }
    cb_wait_front(cb_reduce_scaler, 2);
    cb_pop_front(cb_reduce_scaler, 2);
}
