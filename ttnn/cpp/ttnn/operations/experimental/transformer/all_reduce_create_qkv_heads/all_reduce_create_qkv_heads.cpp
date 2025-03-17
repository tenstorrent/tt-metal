// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_reduce_create_qkv_heads.hpp"

#include "cpp/ttnn/operations/experimental/ccl/reduce_scatter_async/device/reduce_scatter_async_op.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "device/all_reduce_create_qkv_heads_op.hpp"
#include "cpp/ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl::transformer {

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> ExecuteAllReduceCreateQkvHeads::invoke(
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
