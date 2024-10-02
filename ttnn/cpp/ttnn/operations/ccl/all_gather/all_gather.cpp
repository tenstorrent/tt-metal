// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/all_gather/all_gather.hpp"
#include "ttnn/operations/ccl/all_gather/device/all_gather_op.hpp"
#include "ttnn/multi_device.hpp"

namespace ttnn::operations::ccl {

ttnn::Tensor ExecuteAllGather::invoke(const ttnn::Tensor& input_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<size_t> num_workers,
    const std::optional<size_t> num_buffers_per_channel,
    const ttnn::ccl::Topology topology) {
    return ttnn::operations::ccl::all_gather(
        input_tensor, dim, num_links, memory_config, num_workers, num_buffers_per_channel, topology);
}

ttnn::Tensor ExecuteAllGather::invoke(
    const ttnn::Tensor& input_tensor,
    const uint32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<size_t> num_workers,
    const std::optional<size_t> num_buffers_per_channel,
    const ttnn::ccl::Topology topology) {
    return ttnn::operations::ccl::all_gather(
        input_tensor, dim, cluster_axis, mesh_device, num_links, memory_config, num_workers, num_buffers_per_channel, topology);
}

}  // namespace ttnn::operations::ccl
