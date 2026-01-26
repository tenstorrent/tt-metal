// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include <vector>

#include "deepseek_moe_reduce_scatter.hpp"
#include "device/deepseek_moe_reduce_scatter_device_operation.hpp"

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteDeepseekMoEReduceScatter::invoke(
    const std::vector<ttnn::Tensor>& input_tensors,
    const tt::tt_metal::MemoryConfig& output_memory_config,
    int32_t dim,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    std::optional<uint32_t> cluster_axis) {
    uint32_t scatter_dim = (dim < 0) ? dim + input_tensors.at(0).logical_shape().rank() : (uint32_t)dim;

    // topology
    const ttnn::ccl::Topology required_topology = tt::tt_fabric::Topology::Ring;
    ttnn::ccl::Topology usable_topology = ttnn::ccl::get_usable_topology(input_tensors.at(0), topology, cluster_axis);
    usable_topology = ttnn::ccl::convert_2d_to_1d_topology(usable_topology);
    TT_FATAL(
        usable_topology == required_topology,
        "deepseek_moe_reduce_scatter is hardcoded for tt::tt_fabric::Topology::Ring, but has usable_topology {}",
        usable_topology);

    // call the prim operation
    std::vector<ttnn::Tensor> result = ttnn::prim::deepseek_moe_reduce_scatter(
        input_tensors, output_memory_config, scatter_dim, num_links, cluster_axis);

    // return the output tensor (first 8 are intermediates)
    return result.at(8);
}

}  // namespace ttnn::operations::experimental::ccl
