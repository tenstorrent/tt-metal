// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_fabric2d.hpp"
#include "device/combine_fabric2d_device_operation.hpp"
#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

ttnn::Tensor combine_fabric2d(
    const ttnn::Tensor& dispatched_buffer,
    const ttnn::Tensor& dispatched_metadata,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& expert_region_offsets,
    const ttnn::Tensor& expert_offsets,
    uint32_t dispatch_group_size,
    uint32_t experts_per_chip,
    uint32_t num_experts_per_tok,
    uint32_t seq_len_per_chip,
    uint32_t cluster_axis,
    uint32_t num_links,
    std::optional<tt::tt_fabric::Topology> topology,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    bool init_zeros,
    bool use_fp8_combine) {
    // Accepted for signature parity with the production op, but there is nowhere to produce fp8: it comes
    // out of the packer during untilize and this op has no untilize stage.
    TT_FATAL(!use_fp8_combine, "combine_fabric2d does not support fp8 output (no untilize stage to produce it)");
    return ttnn::prim::combine_fabric2d(
        dispatched_buffer.device(),
        dispatched_buffer,
        dispatched_metadata,
        expert_token_counts,
        expert_region_offsets,
        expert_offsets,
        dispatch_group_size,
        experts_per_chip,
        num_experts_per_tok,
        seq_len_per_chip,
        cluster_axis,
        num_links,
        topology.value_or(tt::tt_fabric::Topology::Mesh),
        memory_config.value_or(
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}),
        init_zeros);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
