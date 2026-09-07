// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_fabric2d.hpp"

#include "device/dispatch_fabric2d_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d {

std::array<ttnn::Tensor, 2> dispatch_fabric2d(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& indices_tensor,
    const ttnn::Tensor& expert_offsets_tensor,
    const ttnn::Tensor& expert_dispatch_table_tensor,
    uint32_t experts_per_chip,
    uint32_t num_routed_experts,
    uint32_t num_experts_per_tok,
    uint32_t metadata_len,
    uint32_t max_dispatch_buffer_token_size,
    uint32_t seq_len_per_chip,
    uint32_t cluster_axis,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    const tt::tt_metal::MemoryConfig& memory_config) {
    return ttnn::prim::dispatch_fabric2d(
        input_tensor.device(),
        input_tensor,
        indices_tensor,
        expert_offsets_tensor,
        expert_dispatch_table_tensor,
        experts_per_chip,
        num_routed_experts,
        num_experts_per_tok,
        metadata_len,
        max_dispatch_buffer_token_size,
        seq_len_per_chip,
        cluster_axis,
        num_links,
        topology,
        memory_config);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d
