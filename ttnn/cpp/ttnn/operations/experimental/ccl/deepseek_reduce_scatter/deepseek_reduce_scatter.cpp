// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "deepseek_reduce_scatter.hpp"
#include "device/deepseek_reduce_scatter_device_operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteDeepseekReduceScatter::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::MemoryConfig& output_memory_config,
    uint32_t num_links,
    std::optional<uint32_t> cluster_axis,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id) {
    // Calculate ring size based on cluster_axis
    uint32_t num_devices = ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);
    TT_FATAL(num_devices == 8, "deepseek_reduce_scatter op is hardcoded for 8 devices, but has {}", num_devices);

    ttnn::ccl::Topology usable_topology =
        ttnn::ccl::get_usable_topology(input_tensor, tt::tt_fabric::Topology::Ring, cluster_axis);
    TT_FATAL(
        usable_topology == tt::tt_fabric::Topology::Ring,
        "deepseek_reduce_scatter op is hardcoded for tt::tt_fabric::Topology::Ring, but has usable_topology {}",
        usable_topology);

    // Call the prim operation
    auto result =
        ttnn::prim::deepseek_reduce_scatter(input_tensor, output_memory_config, num_links, cluster_axis, sub_device_id);
    // Return the output tensor (index 1, intermediate is at index 0)
    return result.at(1);
}

}  // namespace ttnn::operations::experimental::ccl
