// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include "ttnn/kernel_lib/host/reduce_host.hpp"

namespace ttnn::prim {

struct SoftmaxReducePlans {
    ttnn::kernel_lib::host::ReduceSequencePlan max;
    ttnn::kernel_lib::host::ReduceSequencePlan sum;

    std::vector<uint32_t> compute_args() const {
        auto args = max.get_compile_time_args();
        sum.append_to(args);
        return args;
    }

    std::vector<uint32_t> auxiliary_args() const {
        auto args = max.get_auxiliary_compile_time_args();
        sum.append_auxiliary_to(args);
        return args;
    }
};

// The fused transforms mask padding before reduction. Streamed reductions
// describe a full pass, an optional repeated pass, and the final short pass.
inline SoftmaxReducePlans make_softmax_reduce_plans(
    uint32_t width_tiles,
    uint32_t pass_tiles,
    tt::tt_metal::DataType max_input_dtype,
    tt::tt_metal::DataType intermediate_dtype,
    const ttnn::kernel_lib::host::ReduceHardwareConfig& hardware,
    compute_kernel_lib::ReduceInputPolicy input_policy) {
    using namespace tt::tt_metal;
    namespace rh = ttnn::kernel_lib::host;
    const uint32_t passes = (width_tiles + pass_tiles - 1) / pass_tiles;
    const uint32_t descriptors = std::min(passes, 3U);
    const TensorLayout max_layout(max_input_dtype, PageConfig(Layout::TILE), MemoryConfig{});
    const TensorLayout intermediate_layout(intermediate_dtype, PageConfig(Layout::TILE), MemoryConfig{});
    const TensorSpec output(Shape{32, 1}, intermediate_layout);
    std::vector<rh::ReduceCbConfig> max_calls;
    std::vector<rh::ReduceCbConfig> sum_calls;
    for (uint32_t i = 0; i < descriptors; ++i) {
        const uint32_t tiles = i + 1 == descriptors ? width_tiles - (passes - 1) * pass_tiles : pass_tiles;
        const Shape shape{32, tiles * 32};
        max_calls.emplace_back(
            0,
            rh::ReduceCallConfig{
                TensorSpec(shape, max_layout), output, ReduceOpMath::MAX, ReduceOpDim::W, 1.0F, ReduceFp32Mode::Fast});
        sum_calls.emplace_back(
            0,
            rh::ReduceCallConfig{
                TensorSpec(shape, intermediate_layout),
                output,
                ReduceOpMath::SUM,
                ReduceOpDim::W,
                1.0F,
                ReduceFp32Mode::Fast});
    }
    SoftmaxReducePlans plans{
        rh::make_reduce_sequence_plan(max_calls, {1, 3, 2}, hardware),
        // BF16 DEST loses small contributions when an unreduced sum is carried
        // over hundreds of tiles. Reduce each pass before cross-call accumulation.
        rh::make_reduce_sequence_plan(
            sum_calls,
            {1, 3, 2},
            hardware,
            passes > 1 && !hardware.fp32_dest_acc_en ? std::optional{compute_kernel_lib::ReduceAlgorithm::ReduceTile}
                                                     : std::nullopt)};
    for (auto* sequence : {&plans.max, &plans.sum}) {
        for (auto& call : sequence->calls) {
            call.plan.input_policy = input_policy;
        }
        if (passes > 1) {
            sequence->calls.back().accumulation_index = passes - 1;
        }
    }
    if (passes > 1 && !hardware.fp32_dest_acc_en) {
        // BF16 needs each pass reduced from zero, then combined by SFPU add.
        // Seeding native reduce with the running scalar sum drops the smaller
        // contributions of individual input tiles as the running sum grows.
        for (auto& call : plans.sum.calls) {
            call.output_cb_id = 2;
            call.accumulator_cb_id = std::nullopt;
            call.accumulation_mode = ttnn::kernel_lib::ReduceAccumulationMode::None;
            call.accumulation_index = 0;
        }
    }
    return plans;
}

}  // namespace ttnn::prim
