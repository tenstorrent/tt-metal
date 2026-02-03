// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "strided_reduce_scatter_async.hpp"
#include "device/strided_reduce_scatter_async_op_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteStridedReduceScatterAsync::invoke(
    const ttnn::Tensor& input_tensor,
    const std::optional<std::vector<ttnn::Tensor>>& persistent_output_buffers,
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::MemoryConfig>& intermediate_memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    std::optional<uint32_t> cluster_axis,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel,
    std::optional<uint32_t> mm_cores_y,
    std::optional<uint32_t> mm_block_ht,
    std::optional<uint32_t> mm_block_wt,
    std::optional<uint32_t> mm_N_block_wt,
    std::optional<uint32_t> chunk_width_in_mm_blocks) {
    int32_t rank = input_tensor.logical_shape().rank();
    int32_t scatter_dim = (dim < 0) ? rank + dim : dim;

    // Calculate ring size based on cluster_axis
    uint32_t num_devices = ::ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);
    TT_FATAL(
        num_devices > 1, "strided_reduce_scatter_async op will only work for num_devices > 1, but has {}", num_devices);

    // Only ring topology is supported for strided reduce scatter
    TT_FATAL(
        topology == ttnn::ccl::Topology::Ring, "strided_reduce_scatter_async currently only supports Ring topology");

    log_debug(tt::LogOp, "strided_reduce_scatter_async: num_devices = {}, topology = Ring", num_devices);

    bool using_persistent_buffers = persistent_output_buffers.has_value();

    std::optional<ttnn::Tensor> optional_intermediate_tensor = std::nullopt;
    std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt;

    if (using_persistent_buffers) {
        const auto& buffers = persistent_output_buffers.value();
        if (!buffers.empty()) {
            optional_intermediate_tensor = buffers[0];
        }
        if (buffers.size() >= 2) {
            optional_output_tensor = buffers[1];
        }
    }

    // Call the prim operation
    auto result = ttnn::prim::strided_reduce_scatter_async(
        input_tensor,
        optional_intermediate_tensor,
        optional_output_tensor,
        scatter_dim,
        num_links,
        num_devices,
        memory_config.value_or(input_tensor.memory_config()),
        intermediate_memory_config,
        topology,
        multi_device_global_semaphore,
        barrier_semaphore,
        using_persistent_buffers,
        sub_device_id,
        cluster_axis,
        chunks_per_sync,
        num_workers_per_link,
        num_buffers_per_channel,
        mm_cores_y,
        mm_block_ht,
        mm_block_wt,
        mm_N_block_wt,
        chunk_width_in_mm_blocks);

    // Return the output tensor (index 1, intermediate is at index 0)
    return result.at(1);
}

}  // namespace ttnn::operations::experimental::ccl
