// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "deepseek_reduce_scatter.hpp"
#include "device/deepseek_reduce_scatter_device_operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteDeepseekReduceScatter::invoke(
    const std::vector<ttnn::Tensor>& input_tensors,
    const ttnn::MemoryConfig& output_memory_config,
    uint32_t num_links,
    std::optional<uint32_t> cluster_axis) {
    // op hardcoded for 8 devices
    uint32_t num_devices = ttnn::ccl::get_topological_dimension(input_tensors.at(0), cluster_axis);
    TT_FATAL(num_devices == 8, "deepseek_reduce_scatter op is hardcoded for 8 devices, but has {}", num_devices);

    // op hardcoded for ring
    ttnn::ccl::Topology usable_topology =
        ttnn::ccl::get_usable_topology(input_tensors.at(0), tt::tt_fabric::Topology::Ring, cluster_axis);
    TT_FATAL(
        usable_topology == tt::tt_fabric::Topology::Ring,
        "deepseek_reduce_scatter op is hardcoded for tt::tt_fabric::Topology::Ring, but has usable_topology {}",
        usable_topology);

    // Call the prim operation
    std::vector<ttnn::Tensor> result =
        ttnn::prim::deepseek_reduce_scatter(input_tensors, output_memory_config, num_links, cluster_axis);

    // Return the output tensor (first 8 are intermediates)
    return result.at(8);
}

}  // namespace ttnn::operations::experimental::ccl
