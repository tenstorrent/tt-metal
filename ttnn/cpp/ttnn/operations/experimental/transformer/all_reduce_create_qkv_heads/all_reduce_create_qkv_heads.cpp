// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_reduce_create_qkv_heads.hpp"

#include "cpp/ttnn/operations/experimental/ccl/reduce_scatter_async/device/reduce_scatter_async_op.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "device/all_reduce_create_qkv_heads_op.hpp"
#include "cpp/ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl::transformer {

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
            tt::log_debug(
                "Found scatter dimension {} for input tensor with padded shape {}", dim, input_tensor_padded_shape);
            return dim;
        }
    }
    TT_THROW(
        "No scatter dimension found for input tensor with padded shape {} and num_workers {}",
        input_tensor_padded_shape,
        num_workers);
}

ttnn::Tensor ExecuteAllReduceCreateQkvHeads::invoke(
    const ttnn::Tensor& input_tensor,
    ttnn::Tensor& buffer_tensor,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt) {
    MemoryConfig out_memory_config = memory_config.value_or(input_tensor.memory_config());
    return ttnn::operations::experimental::ccl::all_reduce_create_qkv_heads(
        input_tensor,
        buffer_tensor,
        cluster_axis,
        mesh_device,
        topology,
        multi_device_global_semaphore,
        out_memory_config,
        num_preferred_links,
        worker_subdevice_id_opt,
        true);
}

}  // namespace ttnn::operations::experimental::ccl::transformer
