// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_fabric2d.hpp"

#include "device/dispatch_fabric2d_device_operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d {

std::array<ttnn::Tensor, 2> dispatch_fabric2d(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& indices_tensor,
    const ttnn::Tensor& expert_offsets_tensor,
    const ttnn::Tensor& expert_dispatch_table_tensor,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& expert_region_offsets,
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
    // Resolve the caller's topology against how this axis is actually wired, the way every other CCL
    // front end does, and store the resolved value. Passing Ring on an axis whose closing link is not
    // wired comes back as Linear, and this op has no Linear mode: it sends single hops around a ring.
    const tt::tt_fabric::Topology usable = ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis);
    TT_FATAL(
        usable == tt::tt_fabric::Topology::Ring || usable == tt::tt_fabric::Topology::Torus,
        "dispatch_fabric2d: axis {} resolves to {}, not a ring. This op relays single hops around one, so "
        "the axis has to be wrap-wired and the topology has to be Ring or Torus; {} was requested.",
        cluster_axis,
        usable,
        topology);

    return ttnn::prim::dispatch_fabric2d(
        input_tensor.device(),
        input_tensor,
        indices_tensor,
        expert_offsets_tensor,
        expert_dispatch_table_tensor,
        expert_token_counts,
        expert_region_offsets,
        experts_per_chip,
        num_routed_experts,
        num_experts_per_tok,
        metadata_len,
        max_dispatch_buffer_token_size,
        seq_len_per_chip,
        cluster_axis,
        num_links,
        usable,
        memory_config);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d
