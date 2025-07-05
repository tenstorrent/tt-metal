// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/decorators.hpp"

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn {
namespace operations {
namespace experimental {
namespace ccl {

struct ExecuteAllReduceAsync {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const GlobalSemaphore& from_remote_multi_device_global_semaphore,
        const GlobalSemaphore& to_remote_multi_device_global_semaphore,
        const GlobalSemaphore& gather_multi_device_global_semaphore,
        ttnn::operations::reduction::ReduceType math_op,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear,
        std::optional<size_t> num_links = std::nullopt,
        std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt = std::nullopt);
    static std::vector<ttnn::Tensor> invoke(
        const std::vector<ttnn::Tensor>& input_tensors,
        const global_semaphore::MultiDeviceGlobalSemaphore& from_remote_multi_device_global_semaphore,
        const global_semaphore::MultiDeviceGlobalSemaphore& to_remote_multi_device_global_semaphore,
        const global_semaphore::MultiDeviceGlobalSemaphore& gather_multi_device_global_semaphore,
        ttnn::operations::reduction::ReduceType math_op,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear,
        std::optional<size_t> num_links = std::nullopt,
        std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt = std::nullopt);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        const GlobalSemaphore& from_remote_multi_device_global_semaphore,
        const GlobalSemaphore& to_remote_multi_device_global_semaphore,
        const GlobalSemaphore& gather_multi_device_global_semaphore,
        ttnn::operations::reduction::ReduceType math_op,
        const std::optional<ttnn::MemoryConfig>& memory_config,
        ttnn::ccl::Topology topology,
        std::optional<size_t> num_preferred_links,
        std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt);
    static std::vector<ttnn::Tensor> invoke(
        const std::vector<ttnn::Tensor>& input_tensors,
        uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        const global_semaphore::MultiDeviceGlobalSemaphore& from_remote_multi_device_global_semaphore,
        const global_semaphore::MultiDeviceGlobalSemaphore& to_remote_multi_device_global_semaphore,
        const global_semaphore::MultiDeviceGlobalSemaphore& gather_multi_device_global_semaphore,
        ttnn::operations::reduction::ReduceType math_op,
        const std::optional<ttnn::MemoryConfig>& memory_config,
        ttnn::ccl::Topology topology,
        std::optional<size_t> num_preferred_links,
        std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        ttnn::Tensor& buffer_tensor,
        uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        const GlobalSemaphore& multi_device_global_semaphore,
        std::optional<const DataType> dtype,
        const std::optional<ttnn::MemoryConfig>& memory_config,
        ttnn::ccl::Topology topology,
        std::optional<size_t> num_preferred_links,
        std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt,
        bool use_noc1_only,
        bool use_optimal_ccl_for_llama);
    static std::vector<ttnn::Tensor> invoke(
        const std::vector<ttnn::Tensor>& input_tensors,
        ttnn::Tensor& buffer_tensor,
        uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
        std::optional<const DataType> dtype,
        const std::optional<ttnn::MemoryConfig>& memory_config,
        ttnn::ccl::Topology topology,
        std::optional<size_t> num_preferred_links,
        std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt,
        bool use_noc1_only,
        bool use_optimal_ccl_for_llama);
};

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

namespace experimental {

constexpr auto all_reduce_async = ttnn::register_operation<
    "ttnn::experimental::all_reduce_async",
    ttnn::operations::experimental::ccl::ExecuteAllReduceAsync>();

}  // namespace experimental
}  // namespace ttnn
