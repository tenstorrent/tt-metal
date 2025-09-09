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
    bool composite_all_gather_case = composite_common::use_composite_all_gather(input_tensor, dim, memory_config);
    bool all_gather_async_llama_sharded_case = composite_common::use_all_gather_async_llama_sharded(
        input_tensor, memory_config.value_or(input_tensor.memory_config()));
    if (composite_all_gather_case && !all_gather_async_llama_sharded_case) {
        return composite_common::composite_all_gather(
            input_tensor,
            dim,
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
            all_gather_async_llama_sharded_case,
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
    bool composite_all_gather_case = composite_common::use_composite_all_gather(input_tensor, dim, memory_config);
    bool all_gather_async_llama_sharded_case = composite_common::use_all_gather_async_llama_sharded(
        input_tensor, memory_config.value_or(input_tensor.memory_config()));
    if (composite_all_gather_case && !all_gather_async_llama_sharded_case) {
        return composite_common::composite_all_gather(
            input_tensor, dim, num_links, memory_config, subdevice_id, cluster_axis);
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
            all_gather_async_llama_sharded_case,
            use_optimal_ccl_for_llama,
            barrier_semaphore,
            chunks_per_sync,
            num_workers_per_link,
            num_buffers_per_channel);
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
    bool composite_all_gather_case = composite_common::use_composite_all_gather(input_tensor, dim, memory_config);
    bool all_gather_async_llama_sharded_case = composite_common::use_all_gather_async_llama_sharded(
        input_tensor, memory_config.value_or(input_tensor.memory_config()));
    if (composite_all_gather_case && !all_gather_async_llama_sharded_case) {
        return composite_common::composite_all_gather(
            input_tensor, dim, num_preferred_links.value_or(1), memory_config, subdevice_id, cluster_axis);
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
            all_gather_async_llama_sharded_case,
            use_optimal_ccl_for_llama,
            barrier_semaphore);
    }
}

}  // namespace ttnn::operations::experimental::ccl
