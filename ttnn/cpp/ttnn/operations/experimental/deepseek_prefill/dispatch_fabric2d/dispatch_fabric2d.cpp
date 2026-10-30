// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_fabric2d.hpp"

#include <tt-metalium/sub_device.hpp>

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
    const std::optional<ttnn::Tensor>& padding_config,
    uint32_t experts_per_chip,
    uint32_t num_routed_experts,
    uint32_t num_experts_per_tok,
    uint32_t metadata_len,
    uint32_t max_dispatch_buffer_token_size,
    uint32_t seq_len_per_chip,
    uint32_t cluster_axis,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    const tt::tt_metal::MemoryConfig& memory_config,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id) {
    // Ring on an axis without a closing link resolves to Linear, which this op cannot run: it sends
    // single hops around a ring.
    const tt::tt_fabric::Topology usable = ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis);
    TT_FATAL(
        usable == tt::tt_fabric::Topology::Ring || usable == tt::tt_fabric::Topology::Torus,
        "dispatch_fabric2d: axis {} resolves to {}, not a ring. The axis must be wrap-wired and the "
        "topology Ring or Torus; {} was requested.",
        cluster_axis,
        usable,
        topology);

    // With no sub-device manager loaded, the first sub-device is the whole compute grid.
    auto* mesh_device = input_tensor.device();
    const auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    const tt::tt_metal::CoreRangeSet allowed_cores =
        mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);

    return ttnn::prim::dispatch_fabric2d(
        input_tensor.device(),
        input_tensor,
        indices_tensor,
        expert_offsets_tensor,
        expert_dispatch_table_tensor,
        expert_token_counts,
        expert_region_offsets,
        padding_config,
        experts_per_chip,
        num_routed_experts,
        num_experts_per_tok,
        metadata_len,
        max_dispatch_buffer_token_size,
        seq_len_per_chip,
        cluster_axis,
        num_links,
        usable,
        memory_config,
        allowed_cores);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d
