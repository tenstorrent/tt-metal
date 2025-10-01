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

    uint32_t num_devices = ::ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);

    int32_t rank = input_tensor.logical_shape().rank();
    int32_t scatter_dim = (dim < 0) ? rank + dim : dim;

    auto output_shape = input_tensor.logical_shape();
    output_shape[scatter_dim] /= num_devices;
    bool is_tiled_and_not_tile_aligned = input_tensor.layout() == Layout::TILE &&
                                         (output_shape[2] % tile_height != 0 || output_shape[3] % tile_width != 0);

    auto input_memory_config = input_tensor.memory_config();
    TT_FATAL(
        !(input_memory_config.is_sharded() && !memory_config.has_value()),
        "If input memory config is sharded, then output memory config must be provided. Defaulting the output memory "
        "config to the input sharded memory config will break the op as the input and output shapes are different.");
    auto output_memory_config = memory_config.value_or(input_memory_config);

    if (input_memory_config.is_sharded()) {
        /*
         * If sharded to interleaved, convert to the final interleaved memory config.
         * If sharded to sharded, use DRAM interleaved as the intermediate memory
         * config for executing the composite.
         */
        auto intermediate_memory_config =
            output_memory_config.is_sharded() ? ttnn::DRAM_MEMORY_CONFIG : output_memory_config;
        input_tensor = ttnn::to_memory_config(input_tensor, intermediate_memory_config);
    }

    std::vector<ttnn::Tensor> broadcasted_tensors = ttnn::operations::experimental::ccl::all_broadcast_async(
        input_tensor, num_links, input_tensor.memory_config(), ttnn::ccl::Topology::Linear, cluster_axis, subdevice_id);

    // Reduce broadcasted tensors into a single reduced tensor
    ttnn::Tensor all_reduced_tensor = broadcasted_tensors[0];
    for (uint32_t i = 1; i < broadcasted_tensors.size(); ++i) {
        all_reduced_tensor = ttnn::add(all_reduced_tensor, broadcasted_tensors[i]);
        broadcasted_tensors[i].deallocate();
    }

    // Convert to row-major (if necessary)
    if (is_tiled_and_not_tile_aligned) {
        // If input is tiled bfloat8_b, cast up to bfloat16 prior to converting to row-major
        if (input_tensor.dtype() == DataType::BFLOAT8_B) {
            all_reduced_tensor = ttnn::typecast(all_reduced_tensor, DataType::BFLOAT16);
        }
        all_reduced_tensor = ttnn::to_layout(all_reduced_tensor, Layout::ROW_MAJOR);
    }

    // Partition the reduced tensor (scatter)
    ttnn::Tensor reduce_scatter_output_tensor =
        ttnn::prim::mesh_partition(all_reduced_tensor, scatter_dim, cluster_axis, all_reduced_tensor.memory_config());

    // Convert back to tiled (if necessary)
    if (is_tiled_and_not_tile_aligned) {
        reduce_scatter_output_tensor = ttnn::to_layout(reduce_scatter_output_tensor, Layout::TILE);
        // If input was tiled bfloat8_b, cast back down to bfloat8_b
        if (input_tensor.dtype() == DataType::BFLOAT8_B) {
            reduce_scatter_output_tensor = ttnn::typecast(reduce_scatter_output_tensor, DataType::BFLOAT8_B);
        }
    }

    if (output_memory_config.is_sharded()) {
        reduce_scatter_output_tensor = ttnn::to_memory_config(reduce_scatter_output_tensor, output_memory_config);
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
    log_debug(tt::LogOp, "DEBUG: using reduce_scatter_minimal_async");
    if (composite_common::use_composite_reduce_scatter(input_tensor, topology, dim, cluster_axis)) {
        log_debug(tt::LogOp, "DEBUG: using composite_reduce_scatter");
        return composite_reduce_scatter(input_tensor, dim, num_links, memory_config, subdevice_id, cluster_axis);
    } else {
        log_debug(tt::LogOp, "DEBUG: using reduce_scatter_minimal_async");
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
