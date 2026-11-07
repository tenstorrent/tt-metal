// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Every core, for each 32-token block:
//   logits = input @ weight, K in one block accumulated in the fp32 DST in K order, as ttnn.linear runs the
//            router with K in one block;
//   rows   = the input tiles untilized, for the dispatch;
//   scores = the logits' softmax over the experts' columns: ttnn.softmax's numerically stable path with its
//            padding mask (one tile a row), through the same compute_kernel_lib helpers and buffer formats.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/softmax.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t hidden_tiles = get_named_compile_time_arg_val("hidden_tiles");
    constexpr uint32_t input_cb = get_named_compile_time_arg_val("input_cb");
    constexpr uint32_t weight_cb = get_named_compile_time_arg_val("weight_cb");
    constexpr uint32_t logits_cb = get_named_compile_time_arg_val("logits_cb");
    constexpr uint32_t mask_cb = get_named_compile_time_arg_val("mask_cb");
    constexpr uint32_t masked_cb = get_named_compile_time_arg_val("masked_cb");
    constexpr uint32_t max_cb = get_named_compile_time_arg_val("max_cb");
    constexpr uint32_t max_scaler_cb = get_named_compile_time_arg_val("max_scaler_cb");
    constexpr uint32_t exps_cb = get_named_compile_time_arg_val("exps_cb");
    constexpr uint32_t sum_scaler_cb = get_named_compile_time_arg_val("sum_scaler_cb");
    constexpr uint32_t recip_sum_cb = get_named_compile_time_arg_val("recip_sum_cb");
    constexpr uint32_t scores_cb = get_named_compile_time_arg_val("scores_cb");
    constexpr uint32_t rows_cb = get_named_compile_time_arg_val("rows_cb");

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    DataflowBuffer input(input_cb);
    DataflowBuffer weight(weight_cb);
    DataflowBuffer logits(logits_cb);
    DataflowBuffer mask(mask_cb);
    DataflowBuffer masked(masked_cb);
    DataflowBuffer max(max_cb);
    DataflowBuffer max_scaler(max_scaler_cb);
    DataflowBuffer exps(exps_cb);
    DataflowBuffer sum_scaler(sum_scaler_cb);
    DataflowBuffer recip_sum(recip_sum_cb);
    DataflowBuffer scores(scores_cb);

    // Matmul takes in0 (the input) on SrcB and in1 (the weight) on SrcA.
    compute_kernel_hw_startup<SrcOrder::Reverse>(input_cb, weight_cb, logits_cb);
    weight.wait_front(hidden_tiles);
    max_scaler.wait_front(1);
    sum_scaler.wait_front(1);

    for (uint32_t block = 0; block < num_blocks; ++block) {
        input.wait_front(hidden_tiles);

        matmul_init(input_cb, weight_cb);
        reconfig_data_format(weight_cb, input_cb);
        pack_reconfig_data_format(logits_cb);
        logits.reserve_back(1);
        tile_regs_acquire();
        for (uint32_t k = 0; k < hidden_tiles; ++k) {
            matmul_tiles(input_cb, weight_cb, k, k, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, logits_cb);
        tile_regs_release();
        logits.push_back(1);

        // Pops the input tiles.
        ckl::untilize<
            hidden_tiles,
            input_cb,
            rows_cb,
            ckl::untilize_config::InitUninitMode::InitAndUninit,
            ckl::untilize_config::WaitMode::NoWait>(1);

        // ttnn.softmax for a row of one tile with its padding mask: the mask added to the tile ...
        reconfig_data_format(logits_cb, logits_cb);
        pack_reconfig_data_format(exps_cb);
        copy_init(logits_cb);
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Add,
                ckl::input(logits_cb),
                ckl::input(mask_cb, ckl::BroadcastDim::Row, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None)>{
                logits, mask},
            ckl::PackTile<ckl::output(
                masked_cb, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled)>{
                masked});

        // ... its row max, exp(x - max) ...
        compute_kernel_lib::reduce<
            PoolType::MAX,
            ReduceDim::REDUCE_ROW,
            masked_cb,
            max_scaler_cb,
            max_cb,
            compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(compute_kernel_lib::ReduceInputBlockShape::row(1));
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(1).block_size(1),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Sub,
                ckl::input(
                    masked_cb,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::AtEnd,
                    ckl::InputTileMapping::Block,
                    ckl::DataFormatReconfig::Disabled),
                ckl::input(max_cb, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd)>{
                masked, max},
            ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(
                exps_cb,
                ckl::ReservePolicy::PerBlockSize,
                ckl::PushPolicy::PerBlockSize,
                ckl::DataFormatReconfig::Disabled)>{exps});
        exps.wait_front(1);

        // ... 1 / its row sum, with the fp32 sum's precision ...
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            exps_cb,
            sum_scaler_cb,
            recip_sum_cb,
            compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
            compute_kernel_lib::ReduceInputBlockShape::row(1),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            [](uint32_t) {
                if constexpr (DST_ACCUM_MODE) {
                    recip_tile_init<ReciprocalDestAcc::FP32, ReciprocalApproxMode::Precise>();
                    recip_tile<ReciprocalDestAcc::FP32, ReciprocalApproxMode::Precise>(0);
                } else {
                    recip_tile_init();
                    recip_tile(0);
                }
            });

        // ... and the exps times it.
        ckl::mul<
            ckl::input(exps_cb, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
            ckl::input(recip_sum_cb, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
            ckl::output(scores_cb, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
            ckl::IterationShape::tiles(1).block_size(1), exps, recip_sum, scores);
    }

    weight.pop_front(hidden_tiles);
    max_scaler.pop_front(1);
    sum_scaler.pop_front(1);
    // The mask is pushed once and waited again every block.
    mask.wait_front(1);
    dummy_unpack(mask_cb);
    mask.pop_front(1);
}
