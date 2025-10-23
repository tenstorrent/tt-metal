// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_reduce.hpp"

#include <tt-metalium/fabric.hpp>

#include "ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"

namespace ttnn::operations::ccl {

ttnn::Tensor ExecuteAllReduce::invoke(
    const ttnn::Tensor& input_tensor,
    std::optional<uint32_t> cluster_axis,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<uint32_t> num_links,
    std::optional<tt::tt_fabric::Topology> topology) {
    // Get mesh device from input tensor
    auto mesh_device = input_tensor.device();
    TT_FATAL(mesh_device != nullptr, "Mesh device is required for all_reduce operation");

    // Determine topology
    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis);

    // Call the experimental all_reduce_async with Sum operation
    return ::ttnn::experimental::all_reduce_async(
        input_tensor,
        cluster_axis,
        *mesh_device,
        std::nullopt,                                  // barrier_semaphores
        std::nullopt,                                  // rs_global_semaphores
        std::nullopt,                                  // ag_global_semaphores
        ttnn::operations::reduction::ReduceType::Sum,  // Always use Sum
        memory_config,
        topology_,
        num_links,
        subdevice_id);
}

}  // namespace ttnn::operations::ccl
