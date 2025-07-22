// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteAllGatherAsync {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        int32_t dim,
        const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
        uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
        bool use_optimal_ccl_for_llama = false);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const std::optional<ttnn::Tensor>& persistent_output_buffer,
        int32_t dim,
        const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
        uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
        std::optional<uint32_t> cluster_axis = std::nullopt,
        bool use_optimal_ccl_for_llama = false);

    static std::vector<ttnn::Tensor> invoke(
        const std::vector<ttnn::Tensor>& input_tensors,
        int32_t dim,
        const std::vector<global_semaphore::MultiDeviceGlobalSemaphore>& multi_device_global_semaphore,
        uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
        bool use_optimal_ccl_for_llama = false);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        int32_t dim,
        uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        ttnn::ccl::Topology topology,
        const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
        const std::optional<ttnn::Tensor>& persistent_output_tensor = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<size_t> num_preferred_links = std::nullopt,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
        bool use_optimal_ccl_for_llama = false);

    static std::vector<ttnn::Tensor> invoke(
        const std::vector<ttnn::Tensor>& input_tensors,
        int32_t dim,
        uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        ttnn::ccl::Topology topology,
        const std::vector<global_semaphore::MultiDeviceGlobalSemaphore>& multi_device_global_semaphore,
        const std::optional<ttnn::Tensor>& persistent_output_tensor = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<size_t> num_preferred_links = std::nullopt,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
        bool use_optimal_ccl_for_llama = false);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto all_gather_async = ttnn::register_operation<
    "ttnn::experimental::all_gather_async",
    ttnn::operations::experimental::ccl::ExecuteAllGatherAsync>();

}  // namespace experimental
}  // namespace ttnn
