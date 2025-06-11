// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_reduce_create_qkv_heads.hpp"

#include "cpp/ttnn/operations/experimental/ccl/reduce_scatter_async/device/reduce_scatter_async_op.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "device/all_reduce_create_qkv_heads_op.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl::transformer {

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> ExecuteAllReduceCreateQkvHeads::invoke(
    const ttnn::Tensor& input_tensor,
    ttnn::Tensor& buffer_tensor,
    const ttnn::Tensor& batch_offset,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const GlobalSemaphore& multi_device_global_semaphore,
    // create qkv heads non-optional parameters
    const uint32_t num_heads,
    const std::optional<ttnn::MemoryConfig>& all_reduce_memory_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt,
    // create qkv heads optional parameters
    std::optional<const uint32_t> num_kv_heads,
    const std::optional<const uint32_t> slice_size,
    const std::optional<MemoryConfig>& final_memory_config,
    const std::optional<const DataType> dtype) {
    MemoryConfig out_memory_config = all_reduce_memory_config.value_or(input_tensor.memory_config());
    const uint32_t num_kv_heads_val = num_kv_heads.value_or(num_heads);
    TT_FATAL(
        input_tensor.padded_shape().size() == 4,
        "Input Tensor dim must be 4 while given dim = {}",
        input_tensor.padded_shape().size());
    TT_FATAL(
        input_tensor.padded_shape()[3] % (num_heads + 2 * num_kv_heads_val) == 0,
        "Input shape {} must be divisible by num_heads + 2*num_kv_heads = {}",
        input_tensor.padded_shape()[3],
        num_heads + 2 * num_kv_heads_val);
    uint32_t head_dim = input_tensor.padded_shape()[3] / (num_heads + 2 * num_kv_heads_val);
    return ttnn::operations::experimental::ccl::all_reduce_create_qkv_heads(
        input_tensor,
        buffer_tensor,
        batch_offset,
        cluster_axis,
        mesh_device,
        topology,
        multi_device_global_semaphore,
        out_memory_config,
        num_preferred_links,
        worker_subdevice_id_opt,
        head_dim,
        num_heads,
        num_kv_heads.value_or(1),
        false,
        slice_size,
        final_memory_config,
        dtype);
}

}  // namespace ttnn::operations::experimental::ccl::transformer
