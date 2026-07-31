// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_fabric2d.hpp"
#include "device/combine_fabric2d_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

ttnn::Tensor combine_fabric2d(
    ttnn::MeshDevice& device,
    const ttnn::Tensor& dispatched_buffer,
    const ttnn::Tensor& dispatched_metadata,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& expert_region_offsets,
    const ttnn::Tensor& expert_offsets,
    uint32_t dispatch_group_size,
    uint32_t experts_per_chip,
    uint32_t num_experts_per_tok,
    uint32_t seq_len_per_chip,
    uint32_t axis,
    uint32_t num_links,
    std::optional<tt::tt_fabric::Topology> topology,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    bool init_zeros,
    uint32_t num_l1_slots,
    uint32_t fwd_bump_every,
    uint32_t assignment_order,
    uint32_t stall_telemetry) {
    return ttnn::prim::combine_fabric2d(
        &device,
        dispatched_buffer,
        dispatched_metadata,
        expert_token_counts,
        expert_region_offsets,
        expert_offsets,
        dispatch_group_size,
        experts_per_chip,
        num_experts_per_tok,
        seq_len_per_chip,
        axis,
        num_links,
        topology.value_or(tt::tt_fabric::Topology::Mesh),
        memory_config.value_or(
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}),
        init_zeros,
        num_l1_slots,
        fwd_bump_every,
        assignment_order,
        stall_telemetry);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
