// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/decorators.hpp"

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn {
namespace operations {
namespace experimental {
namespace ccl {

struct ExecuteReduceScatter {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const int32_t dim,
        const GlobalSemaphore& from_remote_multi_device_global_semaphore,
        const GlobalSemaphore& to_remote_multi_device_global_semaphore,
        ttnn::operations::reduction::ReduceType math_op,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear,
        const std::optional<size_t> num_links = std::nullopt,
        std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt = std::nullopt);
    static std::vector<ttnn::Tensor> invoke(
        const std::vector<ttnn::Tensor>& input_tensors,
        const int32_t dim,
        const global_semaphore::MultiDeviceGlobalSemaphore& from_remote_multi_device_global_semaphore,
        const global_semaphore::MultiDeviceGlobalSemaphore& to_remote_multi_device_global_semaphore,
        ttnn::operations::reduction::ReduceType math_op,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear,
        const std::optional<size_t> num_links = std::nullopt,
        std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt = std::nullopt);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const int32_t dim,
        const uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        const GlobalSemaphore& from_remote_multi_device_global_semaphore,
        const GlobalSemaphore& to_remote_multi_device_global_semaphore,
        const std::optional<std::vector<ttnn::Tensor>>& persistent_output_tensors,
        ttnn::operations::reduction::ReduceType math_op,
        const std::optional<ttnn::MemoryConfig>& memory_config,
        ttnn::ccl::Topology topology,
        const std::optional<size_t> num_preferred_links,
        std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt);
    static std::vector<ttnn::Tensor> invoke(
        const std::vector<ttnn::Tensor>& input_tensors,
        const int32_t dim,
        const uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        const global_semaphore::MultiDeviceGlobalSemaphore& from_remote_multi_device_global_semaphore,
        const global_semaphore::MultiDeviceGlobalSemaphore& to_remote_multi_device_global_semaphore,
        const std::optional<std::vector<ttnn::Tensor>>& persistent_output_tensors,
        ttnn::operations::reduction::ReduceType math_op,
        const std::optional<ttnn::MemoryConfig>& memory_config,
        ttnn::ccl::Topology topology,
        const std::optional<size_t> num_preferred_links,
        std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt);
};

}  // namespace ccl
} // namespace experimental
} // namespace operations

namespace experimental {

constexpr auto reduce_scatter_async = ttnn::register_operation<
    "ttnn::experimental::reduce_scatter_async",
    ttnn::operations::experimental::ccl::ExecuteReduceScatter>();

}  // namespace experimental
}  // namespace ttnn
