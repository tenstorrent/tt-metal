// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_async_generic.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/experimental/ccl/all_to_all_async_generic/device/all_to_all_async_generic_device_operation.hpp"

namespace ttnn::experimental {

ttnn::Tensor all_to_all_async_generic(
    const ttnn::Tensor& input_tensor,
    int32_t in_dim,
    int32_t out_dim,
    const std::optional<Tensor>& persistent_output_buffer,
    std::optional<uint32_t> num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<ttnn::ccl::Topology> topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    std::optional<uint32_t> cluster_axis) {
    auto* mesh_device = input_tensor.device();
    // On FABRIC_2D the sender kernel injects through the routing-table-selected compass connection, which is
    // only well-defined along a single mesh axis; require the caller to name it. (1D infers direction from
    // hop sign and needs no axis.) topology is left to resolve automatically: get_usable_topology maps the
    // active 2D fabric to Mesh, then convert_2d_to_1d_topology collapses it to Linear.
    if (tt::tt_fabric::is_2d_fabric_config(tt::tt_fabric::GetFabricConfig())) {
        TT_FATAL(cluster_axis.has_value(), "all_to_all_async_generic on FABRIC_2D requires a cluster_axis");
    }
    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis);
    topology_ = ::ttnn::ccl::convert_2d_to_1d_topology(topology_);
    uint32_t num_links_ = num_links.value_or(ttnn::operations::ccl::common::get_num_links(*mesh_device, cluster_axis));

    return ttnn::prim::all_to_all_async_generic(
        input_tensor,
        persistent_output_buffer,
        in_dim,
        out_dim,
        num_links_,
        memory_config,
        topology_,
        subdevice_id,
        cluster_axis);
}

}  // namespace ttnn::experimental
