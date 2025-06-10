// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter.hpp"

#include "cpp/ttnn/operations/experimental/ccl/reduce_scatter_async/device/reduce_scatter_async_op.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteReduceScatter::invoke(
    const ttnn::Tensor& input_tensor,
    const int32_t dim,
    const GlobalSemaphore& from_remote_multi_device_global_semaphore,
    const GlobalSemaphore& to_remote_multi_device_global_semaphore,
    ttnn::operations::reduction::ReduceType math_op,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt) {
    MemoryConfig out_memory_config = memory_config.value_or(input_tensor.memory_config());
    return ttnn::operations::experimental::ccl::reduce_scatter(
        input_tensor,
        dim,
        from_remote_multi_device_global_semaphore,
        to_remote_multi_device_global_semaphore,
        math_op,
        out_memory_config,
        topology,
        num_preferred_links,
        worker_subdevice_id_opt);
}

std::vector<ttnn::Tensor> ExecuteReduceScatter::invoke(
    const std::vector<ttnn::Tensor>& input_tensors,
    const int32_t dim,
    const global_semaphore::MultiDeviceGlobalSemaphore& from_remote_multi_device_global_semaphore,
    const global_semaphore::MultiDeviceGlobalSemaphore& to_remote_multi_device_global_semaphore,
    ttnn::operations::reduction::ReduceType math_op,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt) {
    MemoryConfig out_memory_config = memory_config.value_or(input_tensors.at(0).memory_config());
    return ttnn::operations::experimental::ccl::reduce_scatter(
        input_tensors,
        dim,
        from_remote_multi_device_global_semaphore,
        to_remote_multi_device_global_semaphore,
        math_op,
        out_memory_config,
        topology,
        num_preferred_links,
        worker_subdevice_id_opt);
}

ttnn::Tensor ExecuteReduceScatter::invoke(
    const ttnn::Tensor& input_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const GlobalSemaphore& from_remote_multi_device_global_semaphore,
    const GlobalSemaphore& to_remote_multi_device_global_semaphore,
    const std::optional<std::vector<ttnn::Tensor>>& persistent_output_tensors,
    ttnn::operations::reduction::ReduceType math_op,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt) {
    MemoryConfig out_memory_config = memory_config.value_or(input_tensor.memory_config());
    return ttnn::operations::experimental::ccl::reduce_scatter(
        input_tensor,
        dim,
        cluster_axis,
        mesh_device,
        from_remote_multi_device_global_semaphore,
        to_remote_multi_device_global_semaphore,
        persistent_output_tensors,
        math_op,
        out_memory_config,
        topology,
        num_preferred_links,
        worker_subdevice_id_opt);
}

std::vector<ttnn::Tensor> ExecuteReduceScatter::invoke(
    const std::vector<ttnn::Tensor>& input_tensors,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const global_semaphore::MultiDeviceGlobalSemaphore& from_remote_multi_device_global_semaphore,
    const global_semaphore::MultiDeviceGlobalSemaphore& to_remote_multi_device_global_semaphore,
    const std::optional<std::vector<ttnn::Tensor>>& persistent_output_tensors,
    ttnn::operations::reduction::ReduceType math_op,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt) {
    MemoryConfig out_memory_config = memory_config.value_or(input_tensors.at(0).memory_config());
    return ttnn::operations::experimental::ccl::reduce_scatter(
        input_tensors,
        dim,
        cluster_axis,
        mesh_device,
        from_remote_multi_device_global_semaphore,
        to_remote_multi_device_global_semaphore,
        persistent_output_tensors,
        math_op,
        out_memory_config,
        topology,
        num_preferred_links,
        worker_subdevice_id_opt);
}

}  // namespace ttnn::operations::ccl
