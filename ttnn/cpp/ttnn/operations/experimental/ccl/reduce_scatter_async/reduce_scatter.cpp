// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter.hpp"

#include "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_async/device/reduce_scatter_async_op.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteReduceScatter::invoke(
    const ttnn::Tensor& input_tensor,
    const int32_t dim,
    ttnn::operations::reduction::ReduceType math_op,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> num_preferred_links,
    std::optional<SubDeviceId> worker_subdevice_id_opt,
    bool create_semaphore_handles) {
    MemoryConfig out_memory_config = memory_config.value_or(input_tensor.memory_config());
    return ttnn::operations::experimental::ccl::reduce_scatter(
        input_tensor,
        dim,
        math_op,
        out_memory_config,
        topology,
        num_preferred_links,
        worker_subdevice_id_opt,
        create_semaphore_handles);
}

ttnn::Tensor ExecuteReduceScatter::invoke(
    const ttnn::Tensor& input_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::operations::reduction::ReduceType math_op,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> num_preferred_links,
    std::optional<SubDeviceId> worker_subdevice_id_opt,
    bool create_semaphore_handles) {
    MemoryConfig out_memory_config = memory_config.value_or(input_tensor.memory_config());
    return ttnn::operations::experimental::ccl::reduce_scatter(
        input_tensor,
        dim,
        cluster_axis,
        mesh_device,
        math_op,
        out_memory_config,
        topology,
        num_preferred_links,
        worker_subdevice_id_opt,
        create_semaphore_handles);
}

}  // namespace ttnn::operations::ccl
