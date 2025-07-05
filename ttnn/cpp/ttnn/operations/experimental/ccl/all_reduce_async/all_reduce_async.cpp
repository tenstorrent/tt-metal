// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_reduce_async.hpp"

#include "ttnn/operations/experimental/ccl/reduce_scatter_async/device/reduce_scatter_async_op.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "device/all_reduce_async_op.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

uint32_t find_scatter_dim(const ttnn::Shape& input_tensor_padded_shape, size_t num_workers) {
    // iterate until we find a dimension that is divisible by num_workers
    TT_FATAL(input_tensor_padded_shape.size() == 4, "Expected input tensor to have 4 dimensions");
    ttnn::Shape input_tensor_shape_in_tiles{
        input_tensor_padded_shape[0],
        input_tensor_padded_shape[1],
        input_tensor_padded_shape[2] / tt::constants::TILE_HEIGHT,
        input_tensor_padded_shape[3] / tt::constants::TILE_WIDTH};
    for (uint32_t dim = 0; dim < 4; ++dim) {
        if (input_tensor_shape_in_tiles[dim] % num_workers == 0) {
            log_debug(
                tt::LogOp,
                "Found scatter dimension {} for input tensor with padded shape {}",
                dim,
                input_tensor_padded_shape);
            return dim;
        }
    }
    TT_THROW(
        "No scatter dimension found for input tensor with padded shape {} and num_workers {}",
        input_tensor_padded_shape,
        num_workers);
}

ttnn::Tensor ExecuteAllReduceAsync::invoke(
    const ttnn::Tensor& input_tensor,
    const GlobalSemaphore& from_remote_multi_device_global_semaphore,
    const GlobalSemaphore& to_remote_multi_device_global_semaphore,
    const GlobalSemaphore& gather_multi_device_global_semaphore,
    ttnn::operations::reduction::ReduceType math_op,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt) {
    MemoryConfig out_memory_config = memory_config.value_or(input_tensor.memory_config());
    uint32_t dim =
        find_scatter_dim(input_tensor.padded_shape(), ttnn::ccl::get_active_physical_devices(input_tensor).size());
    ttnn::Tensor scattered_tensor = ttnn::operations::experimental::ccl::reduce_scatter(
        input_tensor,
        dim,
        from_remote_multi_device_global_semaphore,
        to_remote_multi_device_global_semaphore,
        math_op,
        out_memory_config,
        topology,
        num_preferred_links,
        worker_subdevice_id_opt);
    return ttnn::operations::experimental::ccl::all_gather_async(
        scattered_tensor,
        dim,
        {gather_multi_device_global_semaphore},
        num_preferred_links.value_or(1),
        out_memory_config,
        topology,
        worker_subdevice_id_opt);
}

std::vector<ttnn::Tensor> ExecuteAllReduceAsync::invoke(
    const std::vector<ttnn::Tensor>& input_tensors,
    const global_semaphore::MultiDeviceGlobalSemaphore& from_remote_multi_device_global_semaphore,
    const global_semaphore::MultiDeviceGlobalSemaphore& to_remote_multi_device_global_semaphore,
    const global_semaphore::MultiDeviceGlobalSemaphore& gather_multi_device_global_semaphore,
    ttnn::operations::reduction::ReduceType math_op,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt) {
    MemoryConfig out_memory_config = memory_config.value_or(input_tensors.at(0).memory_config());
    uint32_t dim = find_scatter_dim(input_tensors.at(0).padded_shape(), input_tensors.size());
    auto scattered_tensors = ttnn::operations::experimental::ccl::reduce_scatter(
        input_tensors,
        dim,
        from_remote_multi_device_global_semaphore,
        to_remote_multi_device_global_semaphore,
        math_op,
        out_memory_config,
        topology,
        num_preferred_links,
        worker_subdevice_id_opt);
    return ttnn::operations::experimental::ccl::all_gather_async(
        scattered_tensors,
        dim,
        {gather_multi_device_global_semaphore},
        num_preferred_links.value_or(1),
        out_memory_config,
        topology,
        worker_subdevice_id_opt);
}

ttnn::Tensor ExecuteAllReduceAsync::invoke(
    const ttnn::Tensor& input_tensor,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const GlobalSemaphore& from_remote_multi_device_global_semaphore,
    const GlobalSemaphore& to_remote_multi_device_global_semaphore,
    const GlobalSemaphore& gather_multi_device_global_semaphore,
    ttnn::operations::reduction::ReduceType math_op,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt) {
    MemoryConfig out_memory_config = memory_config.value_or(input_tensor.memory_config());
    const auto& mesh_view = mesh_device.get_view();
    std::vector<IDevice*> devices =
        (cluster_axis == 0) ? mesh_view.get_devices_on_column(0) : mesh_view.get_devices_on_row(0);
    uint32_t dim = find_scatter_dim(input_tensor.padded_shape(), devices.size());
    ttnn::Tensor scattered_tensor = ttnn::operations::experimental::ccl::reduce_scatter(
        input_tensor,
        dim,
        cluster_axis,
        mesh_device,
        from_remote_multi_device_global_semaphore,
        to_remote_multi_device_global_semaphore,
        std::nullopt,  // persistent_output_tensors
        math_op,
        out_memory_config,
        topology,
        num_preferred_links,
        worker_subdevice_id_opt);
    return ttnn::operations::experimental::ccl::all_gather_async(
        scattered_tensor,
        dim,
        cluster_axis,
        mesh_device,
        topology,
        {gather_multi_device_global_semaphore},
        std::nullopt,  // persistent_output_tensor
        out_memory_config,
        num_preferred_links,
        worker_subdevice_id_opt);
}

std::vector<ttnn::Tensor> ExecuteAllReduceAsync::invoke(
    const std::vector<ttnn::Tensor>& input_tensors,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const global_semaphore::MultiDeviceGlobalSemaphore& from_remote_multi_device_global_semaphore,
    const global_semaphore::MultiDeviceGlobalSemaphore& to_remote_multi_device_global_semaphore,
    const global_semaphore::MultiDeviceGlobalSemaphore& gather_multi_device_global_semaphore,
    ttnn::operations::reduction::ReduceType math_op,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt) {
    MemoryConfig out_memory_config = memory_config.value_or(input_tensors.at(0).memory_config());
    const auto& mesh_view = mesh_device.get_view();
    std::vector<IDevice*> devices =
        (cluster_axis == 0) ? mesh_view.get_devices_on_column(0) : mesh_view.get_devices_on_row(0);
    uint32_t dim = find_scatter_dim(input_tensors.at(0).padded_shape(), devices.size());
    auto scattered_tensors = ttnn::operations::experimental::ccl::reduce_scatter(
        input_tensors,
        dim,
        cluster_axis,
        mesh_device,
        from_remote_multi_device_global_semaphore,
        to_remote_multi_device_global_semaphore,
        std::nullopt,  // persistent_output_tensors
        math_op,
        out_memory_config,
        topology,
        num_preferred_links,
        worker_subdevice_id_opt);
    return ttnn::operations::experimental::ccl::all_gather_async(
        scattered_tensors,
        dim,
        cluster_axis,
        mesh_device,
        topology,
        {gather_multi_device_global_semaphore},
        std::nullopt,  // persistent_output_tensor
        out_memory_config,
        num_preferred_links,
        worker_subdevice_id_opt);
}

ttnn::Tensor ExecuteAllReduceAsync::invoke(
    const ttnn::Tensor& input_tensor,
    ttnn::Tensor& buffer_tensor,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const GlobalSemaphore& multi_device_global_semaphore,
    const std::optional<const DataType> dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt,
    bool use_noc1_only,
    bool use_optimal_ccl_for_llama) {
    MemoryConfig out_memory_config = memory_config.value_or(input_tensor.memory_config());
    return ttnn::operations::experimental::ccl::all_reduce_async(
        input_tensor,
        buffer_tensor,
        cluster_axis,
        mesh_device,
        topology,
        multi_device_global_semaphore,
        dtype,
        out_memory_config,
        num_preferred_links,
        worker_subdevice_id_opt,
        use_noc1_only,
        use_optimal_ccl_for_llama);
}

std::vector<ttnn::Tensor> ExecuteAllReduceAsync::invoke(
    const std::vector<ttnn::Tensor>& input_tensors,
    ttnn::Tensor& buffer_tensor,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
    const std::optional<const DataType> dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt,
    bool use_noc1_only,
    bool use_optimal_ccl_for_llama) {
    MemoryConfig out_memory_config = memory_config.value_or(input_tensors.at(0).memory_config());
    return ttnn::operations::experimental::ccl::all_reduce_async(
        input_tensors,
        buffer_tensor,
        cluster_axis,
        mesh_device,
        topology,
        multi_device_global_semaphore,
        dtype,
        out_memory_config,
        num_preferred_links,
        worker_subdevice_id_opt,
        use_noc1_only,
        use_optimal_ccl_for_llama);
}

}  // namespace ttnn::operations::experimental::ccl
