// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include "ttnn/kernel_lib/host/reduce_host.hpp"

namespace ttnn::operations::moreh {

struct MorehReduceBlocks {
    static constexpr uint32_t tiles_per_block = 8;
    uint32_t buffer_tiles;
    ttnn::kernel_lib::host::ReduceSequencePlan sequence;
};

// Inputs are already masked by the fused transform. Keep the final partial
// block with a full block so the planner can use one accumulation algorithm.
inline MorehReduceBlocks make_moreh_reduce_blocks(
    uint32_t num_tiles,
    tt::tt_metal::ReduceOpDim dim,
    tt::tt_metal::DataType input_dtype,
    tt::tt_metal::DataType output_dtype,
    const ttnn::kernel_lib::host::ReduceHardwareConfig& hardware) {
    using namespace tt::tt_metal;
    namespace rh = ttnn::kernel_lib::host;
    constexpr uint32_t block_tiles = MorehReduceBlocks::tiles_per_block;
    const uint32_t num_blocks = std::max(1U, num_tiles / block_tiles);
    const uint32_t descriptors = std::min(num_blocks, 3U);
    const TensorLayout input_layout(input_dtype, PageConfig(Layout::TILE), MemoryConfig{});
    const TensorLayout output_layout(output_dtype, PageConfig(Layout::TILE), MemoryConfig{});
    std::vector<rh::ReduceCbConfig> calls;
    for (uint32_t i = 0; i < descriptors; ++i) {
        const uint32_t tiles = i + 1 == descriptors ? num_tiles - (num_blocks - 1) * block_tiles : block_tiles;
        const Shape input_shape = dim == ReduceOpDim::W ? Shape{32, tiles * 32} : Shape{tiles * 32, 32};
        const Shape output_shape = dim == ReduceOpDim::W   ? Shape{32, 1}
                                   : dim == ReduceOpDim::H ? Shape{1, 32}
                                                           : Shape{1, 1};
        calls.emplace_back(
            0,
            rh::ReduceCallConfig{
                TensorSpec(input_shape, input_layout),
                TensorSpec(output_shape, output_layout),
                ReduceOpMath::SUM,
                dim,
                1.0F,
                ReduceFp32Mode::Fast});
    }
    auto sequence = rh::make_reduce_sequence_plan(calls, {1, 3, 2}, hardware);
    for (auto& call : sequence.calls) {
        call.plan.input_policy = compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop;
    }
    if (num_blocks > 1) {
        sequence.calls.back().accumulation_index = num_blocks - 1;
    }
    return {std::min(num_tiles, 2 * block_tiles - 1), std::move(sequence)};
}

}  // namespace ttnn::operations::moreh
