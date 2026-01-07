// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_async.hpp"
#include <utility>
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
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
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool reverse_order,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, topology, std::nullopt);
    bool composite_all_gather_case = composite_common::use_composite_all_gather(input_tensor, dim, memory_config);
    bool all_gather_async_llama_sharded_case = composite_common::use_all_gather_async_llama_sharded(
        input_tensor, memory_config.value_or(input_tensor.memory_config()));
    if (composite_all_gather_case && !all_gather_async_llama_sharded_case) {
        log_debug(tt::LogOp, "Using composite_all_gather");
        TT_FATAL(!sub_core_grid.has_value(), "Composite All Gather OP does not currently support sub core grid");
        return composite_common::composite_all_gather(
            input_tensor,
            dim,
            num_links,
            memory_config,
            subdevice_id,
            /*cluster_axis*/ std::nullopt);
    }
    log_debug(tt::LogOp, "Using minimal_all_gather_async");
    return ttnn::operations::experimental::ccl::all_gather_async(
        input_tensor,
        dim,
        multi_device_global_semaphore,
        num_links,
        memory_config,
        topology_,
        subdevice_id,
        use_optimal_ccl_for_llama,
        all_gather_async_llama_sharded_case,
        barrier_semaphore,
        reverse_order,
        sub_core_grid);
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
    std::optional<uint32_t> num_buffers_per_channel,
    bool reverse_order,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis);
    bool composite_all_gather_case = composite_common::use_composite_all_gather(input_tensor, dim, memory_config);
    bool all_gather_async_llama_sharded_case = composite_common::use_all_gather_async_llama_sharded(
        input_tensor, memory_config.value_or(input_tensor.memory_config()));
    if (composite_all_gather_case && !all_gather_async_llama_sharded_case) {
        log_debug(tt::LogOp, "Using composite_all_gather");
        TT_FATAL(!sub_core_grid.has_value(), "Composite All Gather OP does not currently support sub core grid");
        return composite_common::composite_all_gather(
            input_tensor, dim, num_links, memory_config, subdevice_id, cluster_axis);
    }
    log_debug(tt::LogOp, "Using minimal_all_gather_async");
    return ttnn::operations::experimental::ccl::all_gather_async(
        input_tensor,
        persistent_output_buffer,
        dim,
        multi_device_global_semaphore,
        num_links,
        memory_config,
        topology_,
        subdevice_id,
        cluster_axis,
        use_optimal_ccl_for_llama,
        all_gather_async_llama_sharded_case,
        barrier_semaphore,
        chunks_per_sync,
        num_workers_per_link,
        num_buffers_per_channel,
        reverse_order,
        sub_core_grid);
}

std::vector<ttnn::Tensor> ExecuteAllGatherAsync::invoke(
    const std::vector<ttnn::Tensor>& input_tensors,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    const int32_t dim,
    const std::vector<global_semaphore::MultiDeviceGlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    std::optional<uint32_t> cluster_axis,
    bool use_optimal_ccl_for_llama,
    const std::optional<std::vector<GlobalSemaphore>>& barrier_semaphore,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensors.at(0), topology, cluster_axis);
    bool composite_all_gather_case =
        composite_common::use_composite_all_gather(input_tensors.at(0), dim, memory_config);
    bool all_gather_async_llama_sharded_case = composite_common::use_all_gather_async_llama_sharded(
        input_tensors.at(0), memory_config.value_or(input_tensors.at(0).memory_config()));
    if (composite_all_gather_case && !all_gather_async_llama_sharded_case) {
        log_debug(tt::LogOp, "Using composite_all_gather");
        TT_FATAL(!sub_core_grid.has_value(), "Composite All Gather OP does not currently support sub core grid");
        return composite_common::composite_all_gather(
            input_tensors, dim, num_links, memory_config, subdevice_id, cluster_axis);
    }
    log_debug(tt::LogOp, "Using minimal_all_gather_async");
    return ttnn::operations::experimental::ccl::all_gather_async(
        input_tensors,
        persistent_output_buffer,
        dim,
        multi_device_global_semaphore,
        num_links,
        memory_config,
        topology_,
        subdevice_id,
        cluster_axis,
        all_gather_async_llama_sharded_case,
        use_optimal_ccl_for_llama,
        barrier_semaphore,
        chunks_per_sync,
        num_workers_per_link,
        num_buffers_per_channel,
        false,
        sub_core_grid);
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
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool reverse_order,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis);
    bool composite_all_gather_case = composite_common::use_composite_all_gather(input_tensor, dim, memory_config);
    bool all_gather_async_llama_sharded_case = composite_common::use_all_gather_async_llama_sharded(
        input_tensor, memory_config.value_or(input_tensor.memory_config()));
    if (composite_all_gather_case && !all_gather_async_llama_sharded_case) {
        log_debug(tt::LogOp, "Using composite_all_gather");
        TT_FATAL(!sub_core_grid.has_value(), "Composite All Gather OP does not currently support sub core grid");
        return composite_common::composite_all_gather(
            input_tensor, dim, num_preferred_links.value_or(1), memory_config, subdevice_id, cluster_axis);
    }
    log_debug(tt::LogOp, "Using minimal_all_gather_async");
    return ttnn::operations::experimental::ccl::all_gather_async(
        input_tensor,
        dim,
        cluster_axis,
        mesh_device,
        topology_,
        multi_device_global_semaphore,
        persistent_output_tensor,
        memory_config,
        num_preferred_links,
        subdevice_id,
        all_gather_async_llama_sharded_case,
        use_optimal_ccl_for_llama,
        barrier_semaphore,
        reverse_order,
        sub_core_grid);
}

ttnn::Tensor ExecuteAllGatherAsyncReversed::invoke(
    const ttnn::Tensor& input_tensor,
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    bool use_optimal_ccl_for_llama,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool reverse_order,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    // NOTE: reverse_order parameter is ignored, always use true for reversed API
    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, topology, std::nullopt);
    bool composite_all_gather_case = composite_common::use_composite_all_gather(input_tensor, dim, memory_config);
    bool all_gather_async_llama_sharded_case = composite_common::use_all_gather_async_llama_sharded(
        input_tensor, memory_config.value_or(input_tensor.memory_config()));
    if (composite_all_gather_case && !all_gather_async_llama_sharded_case) {
        log_debug(tt::LogOp, "Using composite_all_gather");
        TT_FATAL(!sub_core_grid.has_value(), "Composite All Gather OP does not currently support sub core grid");
        return composite_common::composite_all_gather(
            input_tensor,
            dim,
            num_links,
            memory_config,
            subdevice_id,
            /*cluster_axis*/ std::nullopt);
    }
    log_debug(tt::LogOp, "Using minimal_all_gather_async");
    return ttnn::operations::experimental::ccl::all_gather_async(
        input_tensor,
        dim,
        multi_device_global_semaphore,
        num_links,
        memory_config,
        topology_,
        subdevice_id,
        use_optimal_ccl_for_llama,
        all_gather_async_llama_sharded_case,
        barrier_semaphore,
        true,
        sub_core_grid);  // reverse_order=true for reversed API
}

ttnn::Tensor ExecuteAllGatherAsyncReversed::invoke(
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
    std::optional<uint32_t> num_buffers_per_channel,
    bool reverse_order,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis);
    bool composite_all_gather_case = composite_common::use_composite_all_gather(input_tensor, dim, memory_config);
    bool all_gather_async_llama_sharded_case = composite_common::use_all_gather_async_llama_sharded(
        input_tensor, memory_config.value_or(input_tensor.memory_config()));
    if (composite_all_gather_case && !all_gather_async_llama_sharded_case) {
        log_debug(tt::LogOp, "Using composite_all_gather");
        TT_FATAL(!sub_core_grid.has_value(), "Composite All Gather OP does not currently support sub core grid");
        return composite_common::composite_all_gather(
            input_tensor, dim, num_links, memory_config, subdevice_id, cluster_axis);
    }
    log_debug(tt::LogOp, "Using minimal_all_gather_async");
    return ttnn::operations::experimental::ccl::all_gather_async(
        input_tensor,
        persistent_output_buffer,
        dim,
        multi_device_global_semaphore,
        num_links,
        memory_config,
        topology_,
        subdevice_id,
        cluster_axis,
        use_optimal_ccl_for_llama,
        all_gather_async_llama_sharded_case,
        barrier_semaphore,
        chunks_per_sync,
        num_workers_per_link,
        num_buffers_per_channel,
        true,
        sub_core_grid);  // reverse_order=true for reversed API
}

ttnn::Tensor ExecuteAllGatherAsyncReversed::invoke(
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
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool reverse_order,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis);
    bool composite_all_gather_case = composite_common::use_composite_all_gather(input_tensor, dim, memory_config);
    bool all_gather_async_llama_sharded_case = composite_common::use_all_gather_async_llama_sharded(
        input_tensor, memory_config.value_or(input_tensor.memory_config()));
    if (composite_all_gather_case && !all_gather_async_llama_sharded_case) {
        log_debug(tt::LogOp, "Using composite_all_gather");
        TT_FATAL(!sub_core_grid.has_value(), "Composite All Gather OP does not currently support sub core grid");
        return composite_common::composite_all_gather(
            input_tensor, dim, num_preferred_links.value_or(1), memory_config, subdevice_id, cluster_axis);
    }
    log_debug(tt::LogOp, "Using minimal_all_gather_async");
    return ttnn::operations::experimental::ccl::all_gather_async(
        input_tensor,
        dim,
        cluster_axis,
        mesh_device,
        topology_,
        multi_device_global_semaphore,
        persistent_output_tensor,
        memory_config,
        num_preferred_links,
        subdevice_id,
        all_gather_async_llama_sharded_case,
        use_optimal_ccl_for_llama,
        barrier_semaphore,
        true,
        sub_core_grid);  // reverse_order=true for reversed API
}

}  // namespace ttnn::operations::experimental::ccl
