// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_async.hpp"
#include <utility>
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "ttnn/operations/experimental/ccl/all_broadcast_async/device/all_broadcast_async_op.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

bool use_composite_all_gather(
    const ttnn::Tensor& input_tensor, const int32_t dim, const std::optional<GlobalSemaphore>& barrier_semaphore) {
    auto tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    uint32_t tile_height = tile_shape[0];
    uint32_t tile_width = tile_shape[1];

    int32_t rank = input_tensor.logical_shape().rank();
    int32_t gather_dim = (dim < 0) ? rank + dim : dim;

    // Composite only supported when barrier_semaphore is provided
    if (!barrier_semaphore.has_value()) {
        return false;
    }

    // Use composite for row-major tensors
    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        return true;
    }

    // Use composite if tiled and padded on the gather dim
    auto input_shape = input_tensor.logical_shape();
    if ((gather_dim == 2 && input_shape[2] % tile_height != 0) ||
        (gather_dim == 3 && input_shape[3] % tile_width != 0)) {
        return true;
    }

    return false;
}

// Composite always runs in row-major
ttnn::Tensor composite_all_gather(
    ttnn::Tensor input_tensor,
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const GlobalSemaphore& barrier_semaphore,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    std::optional<uint32_t> cluster_axis) {
    DataType input_dtype = input_tensor.dtype();
    bool convert_to_bfloat16_for_composite = input_dtype == DataType::BFLOAT8_B;
    bool is_tiled = input_tensor.layout() == Layout::TILE;

    int32_t rank = input_tensor.logical_shape().rank();
    int32_t gather_dim = (dim < 0) ? rank + dim : dim;

    // Convert to row major
    if (is_tiled) {
        // If input is tiled bfloat8_b, convert to bfloat16 to do the all_broadcast_async + concat
        if (convert_to_bfloat16_for_composite) {
            input_tensor = ttnn::typecast(input_tensor, DataType::BFLOAT16);
        }
        input_tensor = ttnn::to_layout(input_tensor, Layout::ROW_MAJOR);
    }

    std::vector<ttnn::Tensor> broadcasted_tensors = ttnn::operations::experimental::ccl::all_broadcast_async(
        input_tensor,
        multi_device_global_semaphore[0],
        barrier_semaphore,
        num_links,
        memory_config,
        ttnn::ccl::Topology::Linear,
        cluster_axis,
        subdevice_id);

    ttnn::Tensor all_gather_output_tensor = ttnn::concat(broadcasted_tensors, gather_dim);

    // Convert back to tiled
    if (is_tiled) {
        all_gather_output_tensor = ttnn::to_layout(all_gather_output_tensor, Layout::TILE);
        // If we had to convert the input dtype in order to execute the row-major composite op, convert back to the
        // input dtype
        if (convert_to_bfloat16_for_composite) {
            all_gather_output_tensor = ttnn::typecast(all_gather_output_tensor, input_dtype);
        }
    }

    return all_gather_output_tensor;
}

// Composite always runs in row-major
std::vector<ttnn::Tensor> composite_all_gather(
    const std::vector<ttnn::Tensor>& input_tensors,
    const int32_t dim,
    const std::vector<global_semaphore::MultiDeviceGlobalSemaphore>& multi_device_global_semaphore,
    const GlobalSemaphore& barrier_semaphore,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    std::optional<uint32_t> cluster_axis) {
    std::vector<GlobalSemaphore> semaphore;
    semaphore.reserve(multi_device_global_semaphore.size());
    for (size_t i = 0; i < multi_device_global_semaphore.size(); i++) {
        semaphore.push_back(multi_device_global_semaphore.at(i).global_semaphores.at(i));
    }
    std::vector<Tensor> output_tensors;
    output_tensors.reserve(input_tensors.size());
    for (size_t i = 0; i < input_tensors.size(); i++) {
        output_tensors.push_back(composite_all_gather(
            input_tensors[i], dim, semaphore, barrier_semaphore, num_links, memory_config, subdevice_id, cluster_axis));
    }
    return output_tensors;
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
    if (use_composite_all_gather(input_tensor, dim, barrier_semaphore)) {
        return composite_all_gather(
            input_tensor,
            dim,
            multi_device_global_semaphore,
            barrier_semaphore.value(),
            num_links,
            memory_config,
            subdevice_id,
            /*cluster_axis*/ std::nullopt);
    } else {
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
    if (use_composite_all_gather(input_tensor, dim, barrier_semaphore)) {
        return composite_all_gather(
            input_tensor,
            dim,
            multi_device_global_semaphore,
            barrier_semaphore.value(),
            num_links,
            memory_config,
            subdevice_id,
            cluster_axis);
    } else {
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
    if (use_composite_all_gather(input_tensors[0], dim, barrier_semaphore)) {
        return composite_all_gather(
            input_tensors,
            dim,
            multi_device_global_semaphore,
            barrier_semaphore.value(),
            num_links,
            memory_config,
            subdevice_id,
            /*cluster_axis*/ std::nullopt);
    } else {
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
    if (use_composite_all_gather(input_tensor, dim, barrier_semaphore)) {
        return composite_all_gather(
            input_tensor,
            dim,
            multi_device_global_semaphore,
            barrier_semaphore.value(),
            num_preferred_links.value_or(1),
            memory_config,
            subdevice_id,
            cluster_axis);
    } else {
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
    if (use_composite_all_gather(input_tensors[0], dim, barrier_semaphore)) {
        return composite_all_gather(
            input_tensors,
            dim,
            multi_device_global_semaphore,
            barrier_semaphore.value(),
            num_preferred_links.value_or(1),
            memory_config,
            subdevice_id,
            cluster_axis);
    } else {
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
}

}  // namespace ttnn::operations::experimental::ccl
