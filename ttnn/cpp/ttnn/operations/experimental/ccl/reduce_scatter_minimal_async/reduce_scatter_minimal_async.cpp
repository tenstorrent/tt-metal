// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter_minimal_async.hpp"
#include <utility>
#include "ttnn/operations/ccl/mesh_partition/device/mesh_partition_device_operation.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/experimental/ccl/all_broadcast_async/device/all_broadcast_async_op.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

// Composite always runs in row-major
ttnn::Tensor composite_reduce_scatter(
    ttnn::Tensor input_tensor,
    const int32_t dim,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    std::optional<uint32_t> cluster_axis) {
    auto tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    uint32_t tile_height = tile_shape[0];
    uint32_t tile_width = tile_shape[1];

    uint32_t num_devices;
    if (cluster_axis.has_value()) {
        auto mesh_device = input_tensor.device();
        const auto& mesh_view = mesh_device->get_view();
        num_devices = (cluster_axis.value() == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    } else {
        num_devices = ttnn::ccl::get_active_physical_devices(input_tensor).size();
    }

    auto input_shape = input_tensor.logical_shape();

    int32_t rank = input_tensor.logical_shape().rank();
    int32_t scatter_dim = (dim < 0) ? rank + dim : dim;

    auto output_shape = input_shape;
    output_shape[scatter_dim] /= num_devices;

    bool is_tiled_and_not_tile_aligned = input_tensor.layout() == Layout::TILE &&
                                         (output_shape[2] % tile_height != 0 || output_shape[3] % tile_width != 0);

    // If we need to convert to row-major, then if the input dtype is bfloat8_b we need to typecast before untilizing
    // and after re-tilizing
    DataType input_dtype = input_tensor.dtype();
    bool convert_to_bfloat16_for_composite = is_tiled_and_not_tile_aligned && input_dtype == DataType::BFLOAT8_B;

    // Convert to row major
    if (is_tiled_and_not_tile_aligned) {
        // If input is tiled bfloat8_b, convert to bfloat16 to do the all_broadcast_async + concat
        if (convert_to_bfloat16_for_composite) {
            input_tensor = ttnn::typecast(input_tensor, DataType::BFLOAT16);
        }
        input_tensor = ttnn::to_layout(input_tensor, Layout::ROW_MAJOR);
    }

    // Broadcast each tensor to all other devices in the mesh
    std::vector<ttnn::Tensor> broadcasted_tensors = ttnn::operations::experimental::ccl::all_broadcast_async(
        input_tensor, num_links, memory_config, ttnn::ccl::Topology::Linear, cluster_axis, subdevice_id);

    // Reduce broadcasted tensors into a single reduced tensor
    ttnn::Tensor all_reduced_tensor = broadcasted_tensors[0];
    for (uint32_t i = 1; i < broadcasted_tensors.size(); ++i) {
        all_reduced_tensor = ttnn::add(all_reduced_tensor, broadcasted_tensors[i]);
        broadcasted_tensors[i].deallocate();
    }

    // Partition the reduced tensor (scatter)
    ttnn::Tensor reduce_scatter_output_tensor = ttnn::prim::mesh_partition(
        all_reduced_tensor, scatter_dim, cluster_axis, memory_config.value_or(all_reduced_tensor.memory_config()));

    // Convert back to tiled
    if (is_tiled_and_not_tile_aligned) {
        reduce_scatter_output_tensor = ttnn::to_layout(reduce_scatter_output_tensor, Layout::TILE);
        // If we had to convert the input dtype in order to execute the row-major composite op, convert back to the
        // input dtype
        if (convert_to_bfloat16_for_composite) {
            reduce_scatter_output_tensor = ttnn::typecast(reduce_scatter_output_tensor, input_dtype);
        }
    }

    return reduce_scatter_output_tensor;
}

ttnn::Tensor ExecuteReduceScatterMinimalAsync::invoke(
    const ttnn::Tensor& input_tensor,
    const std::optional<std::vector<ttnn::Tensor>>& persistent_output_buffers,
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::MemoryConfig>& intermediate_memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    std::optional<uint32_t> cluster_axis,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel) {
    if (composite_common::use_composite_reduce_scatter(input_tensor, dim, cluster_axis)) {
        return composite_reduce_scatter(input_tensor, dim, num_links, memory_config, subdevice_id, cluster_axis);
    } else {
        return ttnn::operations::experimental::ccl::reduce_scatter_minimal_async(
            input_tensor,
            persistent_output_buffers,
            dim,
            multi_device_global_semaphore,
            barrier_semaphore,
            num_links,
            memory_config,
            intermediate_memory_config,
            topology,
            subdevice_id,
            cluster_axis,
            chunks_per_sync,
            num_workers_per_link,
            num_buffers_per_channel);
    }
}
}  // namespace ttnn::operations::experimental::ccl
