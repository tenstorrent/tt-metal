// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_async.hpp"
#include <utility>
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "ttnn/operations/experimental/ccl/all_broadcast_async/device/all_broadcast_async_op.hpp"
// #include "ttnn/operations/core/to_layout/to_layout_op.cpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

bool use_native_all_gather(
    const ttnn::Tensor& input_tensor, const std::optional<ttnn::MemoryConfig>& memory_config, const uint32_t dim) {
    // Row major
    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        return false;
    }

    // Tiled and padded on the gather dim
    if (dim == 2 || dim == 3) {
        auto input_shape = input_tensor.logical_shape();
        if (input_shape[dim] % 32 != 0) {
            return false;
        }
    }

    return true;
}

ttnn::Tensor all_broadcast_composite_all_gather(
    const ttnn::Tensor& input_tensor,
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    std::optional<uint32_t> cluster_axis,
    const std::optional<GlobalSemaphore>& barrier_semaphore) {
    TT_FATAL(
        barrier_semaphore.has_value(),
        "all_gather_async composite using all_broadcast_async requires a barrier semaphore");

    ttnn::MemoryConfig output_memory_config = memory_config.value_or(input_tensor.memory_config());
    bool is_tiled = input_tensor.layout() == Layout::TILE;

    // Convert to row major
    ttnn::Tensor all_broadcast_input_tensor =
        is_tiled ? ttnn::to_layout(input_tensor, Layout::ROW_MAJOR) : input_tensor;

    ttnn::MemoryConfig all_broadcast_memory_config = input_tensor.memory_config();
    std::vector<ttnn::Tensor> broadcasted_tensors = ttnn::operations::experimental::ccl::all_broadcast_async(
        input_tensor,
        multi_device_global_semaphore[0],
        num_links,
        all_broadcast_memory_config,
        ttnn::ccl::Topology::Linear,
        subdevice_id,
        barrier_semaphore);

    ttnn::Tensor all_gather_output_tensor = ttnn::concat(broadcasted_tensors, dim);

    // Convert to tiled
    if (is_tiled) {
        all_gather_output_tensor = ttnn::to_layout(all_gather_output_tensor, Layout::TILE);
    }

    return all_gather_output_tensor;
}

ttnn::Tensor ExecuteAllGatherAsync::invoke(
    const ttnn::Tensor& input_tensor,
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    bool use_optimal_ccl_for_llama,
    const std::optional<GlobalSemaphore>& barrier_semaphore) {
    std::cout << "A" << std::endl;
    return ttnn::operations::experimental::ccl::all_gather_async(
        input_tensor,
        dim,
        multi_device_global_semaphore,
        num_links,
        memory_config,
        topology,
        subdevice_id,
        use_optimal_ccl_for_llama,
        barrier_semaphore);
}

ttnn::Tensor ExecuteAllGatherAsync::invoke(
    const ttnn::Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    std::optional<uint32_t> cluster_axis,
    bool use_optimal_ccl_for_llama,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel) {
    return all_broadcast_composite_all_gather(
        input_tensor,
        dim,
        multi_device_global_semaphore,
        num_links,
        memory_config,
        subdevice_id,
        cluster_axis,
        barrier_semaphore);

    std::cout << "B" << std::endl;
    return ttnn::operations::experimental::ccl::all_gather_async(
        input_tensor,
        persistent_output_buffer,
        dim,
        multi_device_global_semaphore,
        num_links,
        memory_config,
        topology,
        subdevice_id,
        cluster_axis,
        use_optimal_ccl_for_llama,
        barrier_semaphore,
        chunks_per_sync,
        num_workers_per_link,
        num_buffers_per_channel);
}

std::vector<ttnn::Tensor> ExecuteAllGatherAsync::invoke(
    const std::vector<ttnn::Tensor>& input_tensors,
    const int32_t dim,
    const std::vector<global_semaphore::MultiDeviceGlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    bool use_optimal_ccl_for_llama,
    const std::optional<GlobalSemaphore>& barrier_semaphore) {
    std::cout << "C" << std::endl;
    return ttnn::operations::experimental::ccl::all_gather_async(
        input_tensors,
        dim,
        multi_device_global_semaphore,
        num_links,
        memory_config,
        topology,
        subdevice_id,
        use_optimal_ccl_for_llama,
        barrier_semaphore);
}

ttnn::Tensor ExecuteAllGatherAsync::invoke(
    const ttnn::Tensor& input_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<ttnn::Tensor>& persistent_output_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    bool use_optimal_ccl_for_llama,
    const std::optional<GlobalSemaphore>& barrier_semaphore) {
    std::cout << "D" << std::endl;
    return ttnn::operations::experimental::ccl::all_gather_async(
        input_tensor,
        dim,
        cluster_axis,
        mesh_device,
        topology,
        multi_device_global_semaphore,
        persistent_output_tensor,
        memory_config,
        num_preferred_links,
        subdevice_id,
        use_optimal_ccl_for_llama,
        barrier_semaphore);
}

std::vector<ttnn::Tensor> ExecuteAllGatherAsync::invoke(
    const std::vector<ttnn::Tensor>& input_tensors,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const std::vector<global_semaphore::MultiDeviceGlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<ttnn::Tensor>& persistent_output_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    bool use_optimal_ccl_for_llama,
    const std::optional<GlobalSemaphore>& barrier_semaphore) {
    std::cout << "E" << std::endl;
    return ttnn::operations::experimental::ccl::all_gather_async(
        input_tensors,
        dim,
        cluster_axis,
        mesh_device,
        topology,
        multi_device_global_semaphore,
        persistent_output_tensor,
        memory_config,
        num_preferred_links,
        subdevice_id,
        use_optimal_ccl_for_llama,
        barrier_semaphore);
}

}  // namespace ttnn::operations::experimental::ccl
