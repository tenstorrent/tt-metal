// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_fabric2d.hpp"
#include "device/combine_fabric2d_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

ttnn::Tensor combine_fabric2d(
    const ttnn::Tensor& dispatched_buffer,
    const ttnn::Tensor& dispatched_metadata,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& expert_region_offsets,
    const ttnn::Tensor& expert_offsets,
    uint32_t experts_per_chip,
    uint32_t num_experts_per_tok,
    uint32_t seq_len_per_chip,
    uint32_t cluster_axis,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    const tt::tt_metal::MemoryConfig& memory_config) {
    return ttnn::prim::combine_fabric2d(
        dispatched_buffer.device(),
        dispatched_buffer,
        dispatched_metadata,
        expert_token_counts,
        expert_region_offsets,
        expert_offsets,
        experts_per_chip,
        num_experts_per_tok,
        seq_len_per_chip,
        cluster_axis,
        num_links,
        topology,
        memory_config);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
