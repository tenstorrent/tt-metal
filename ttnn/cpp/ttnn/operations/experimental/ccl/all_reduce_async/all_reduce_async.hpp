// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::experimental {

ttnn::Tensor all_reduce_async(
    const ttnn::Tensor& input_tensor,
    uint32_t num_devices,
    const std::vector<GlobalSemaphore>& barrier_semaphores,
    const std::vector<GlobalSemaphore>& rs_global_semaphores,
    const std::vector<GlobalSemaphore>& ag_global_semaphores,
    ttnn::operations::reduction::ReduceType math_op,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear,
    std::optional<size_t> num_preferred_links = std::nullopt,
    std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt = std::nullopt);

ttnn::Tensor all_reduce_async(
    const ttnn::Tensor& input_tensor,
    std::optional<std::uint32_t> cluster_axis,
    MeshDevice& mesh_device,
    const std::optional<std::vector<GlobalSemaphore>>& barrier_semaphores,
    const std::optional<std::vector<GlobalSemaphore>>& rs_global_semaphores,
    const std::optional<std::vector<GlobalSemaphore>>& ag_global_semaphores,
    ttnn::operations::reduction::ReduceType math_op,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<ttnn::ccl::Topology> topology,
    std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt);

ttnn::Tensor all_reduce_async(
    const ttnn::Tensor& input_tensor,
    uint32_t cluster_axis,
    ttnn::operations::reduction::ReduceType math_op,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    std::optional<size_t> num_preferred_links = std::nullopt,
    std::optional<ttnn::ccl::Topology> topology = std::nullopt);

ttnn::Tensor all_reduce_async(
    const ttnn::Tensor& input_tensor,
    ttnn::Tensor& buffer_tensor,
    uint32_t cluster_axis,
    MeshDevice& mesh_device,
    const GlobalSemaphore& multi_device_global_semaphore,
    std::optional<const DataType> dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt,
    bool use_noc1_only,
    bool use_optimal_ccl_for_llama);

std::vector<ttnn::Tensor> all_reduce_async(
    const std::vector<ttnn::Tensor>& input_tensors,
    ttnn::Tensor& buffer_tensor,
    uint32_t cluster_axis,
    MeshDevice& mesh_device,
    const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
    std::optional<const DataType> dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt,
    bool use_noc1_only,
    bool use_optimal_ccl_for_llama);

}  // namespace ttnn::experimental
