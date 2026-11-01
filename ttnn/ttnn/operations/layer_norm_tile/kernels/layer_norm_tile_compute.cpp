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
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace {

using compute_kernel_lib::DataFormatReconfig;

// Streaming-pass operands: one CB handshake per IterationShape::block_size group of DEST lanes.
constexpr compute_kernel_lib::InputSpec lane_input(
    uint32_t cb, DataFormatReconfig reconfig = DataFormatReconfig::Enabled) {
    using namespace compute_kernel_lib;
    return input(cb, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block, reconfig);
}

constexpr compute_kernel_lib::OutputSpec lane_output(
    uint32_t cb, DataFormatReconfig reconfig = DataFormatReconfig::Enabled) {
    using namespace compute_kernel_lib;
    return output(cb, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize, reconfig);
}

}  // namespace

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
    // Tiles per DEST acquire on the streaming passes; the chain clamps it to DEST capacity.
    constexpr uint32_t dest_lanes = get_compile_time_arg_val(16);

    const uint32_t core_row_tiles = get_arg_val<uint32_t>(0);

    using namespace compute_kernel_lib;

    // Reconfig is dropped only where the format cannot change: work_a / work_b alias one allocation
    // with one page format, and gamma / beta share the I/O format (l1_ledger.md).
    constexpr auto same_work_fmt = DataFormatReconfig::Disabled;

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

    // The REDUCE_ROW result lives in column 0 (faces 0 and 2), and normalize reads only that column
    // through its col broadcast, so the SFPU post-op runs on VectorMode::C: half the faces.
    auto finalize_inv_std = [=](uint32_t dst_idx) {
        binop_with_scalar_tile_init();
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_binop_with_scalar,
            (APPROX, MUL_UNARY, 8, DST_ACCUM_MODE),
            dst_idx,
            VectorMode::C,
            inv_w_bits));
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_binop_with_scalar,
            (APPROX, ADD_UNARY, 8, DST_ACCUM_MODE),
            dst_idx,
            VectorMode::C,
            epsilon_bits));
        rsqrt_tile_init();
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_rsqrt,
            (APPROX, 8, DST_ACCUM_MODE, false),
            dst_idx,
            VectorMode::C));
    };

    for (uint32_t block_row = 0; block_row < core_row_tiles; block_row += block_row_tiles) {
        const uint32_t rows =
            (core_row_tiles - block_row) < block_row_tiles ? (core_row_tiles - block_row) : block_row_tiles;
        const auto block_shape = IterationShape::grid(rows, tensor_w_tiles);
        // Streaming passes take `dest_lanes` tiles per DEST acquire (one CB handshake per group).
        const auto lane_shape = IterationShape::grid(rows, tensor_w_tiles).block_size(dest_lanes);

        // residual_add_block
        if constexpr (has_residual) {
            MaybeDeviceZoneScope("compute_residual_add");
            add<lane_input(cb_input_tiles), lane_input(cb_residual_tiles), lane_output(cb_row_work_a)>(
                lane_shape, input_tiles, residual_tiles, work_a);
        }

        // row_mean_block (source left resident for center_block)
        {
            MaybeDeviceZoneScope("compute_wait_z");
            z_tiles.wait_front(rows * tensor_w_tiles);
        }
        {
            MaybeDeviceZoneScope("compute_row_mean");
            reduce_mean<
                ReduceDim::REDUCE_ROW,
                cb_z,
                cb_gamma_tiles,
                cb_row_mean,
                ReduceInputPolicy::WaitUpfrontNoPop,
                ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                ReduceFp32Mode::Fast,
                ReduceAlgorithm::AccumulateViaAdd>(ReduceInputBlockShape::of(rows, tensor_w_tiles), width);
        }

        // center_block
        {
            MaybeDeviceZoneScope("compute_center");
            sub<lane_input(cb_z),
                input(cb_row_mean, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
                lane_output(cb_row_work_b)>(lane_shape, z_tiles, row_mean, work_b);
        }

        // row_sumsq_block (c stays resident for normalize_block)
        {
            MaybeDeviceZoneScope("compute_row_sumsq");
            work_b.wait_front(rows * tensor_w_tiles);
            eltwise_chain(block_shape, SquareAccumulate{work_b, work_b}, SumsqPack{row_sumsq});
        }

        // row_inv_std_block
        {
            MaybeDeviceZoneScope("compute_inv_std");
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
        }

        // normalize_block
        {
            MaybeDeviceZoneScope("compute_normalize");
            mul<lane_input(cb_row_work_b),
                input(cb_inv_std, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
                lane_output(cb_row_work_a)>(lane_shape, work_b, inv_std, work_a);
        }
        {
            MaybeDeviceZoneScope("compute_wait_affine");
            gamma_tiles.wait_front(tensor_w_tiles);
            beta_tiles.wait_front(tensor_w_tiles);
        }

        // affine_scale_block
        {
            MaybeDeviceZoneScope("compute_affine_scale");
            mul<lane_input(cb_row_work_a, same_work_fmt),
                input(cb_gamma_tiles, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row),
                lane_output(cb_row_work_b, same_work_fmt)>(lane_shape, work_a, gamma_tiles, work_b);
        }

        // affine_shift_block
        {
            MaybeDeviceZoneScope("compute_affine_shift");
            add<lane_input(cb_row_work_b, same_work_fmt),
                input(
                    cb_beta_tiles,
                    BroadcastDim::Row,
                    WaitPolicy::Upfront,
                    PopPolicy::None,
                    InputTileMapping::Row,
                    same_work_fmt),
                lane_output(cb_output_tiles)>(lane_shape, work_b, beta_tiles, output_tiles);
        }
    }

    gamma_tiles.pop_front(tensor_w_tiles);
    beta_tiles.pop_front(tensor_w_tiles);
}
