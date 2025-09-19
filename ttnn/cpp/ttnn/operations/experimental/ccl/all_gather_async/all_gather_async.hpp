// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/experimental/ccl/common.hpp"

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
        bool use_optimal_ccl_for_llama = false,
        const std::optional<GlobalSemaphore>& barrier_semaphore = std::nullopt,
        bool reverse_order = false);

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
        bool use_optimal_ccl_for_llama = false,
        const std::optional<GlobalSemaphore>& barrier_semaphore = std::nullopt,
        std::optional<uint32_t> chunks_per_sync = std::nullopt,
        std::optional<uint32_t> num_workers_per_link = std::nullopt,
        std::optional<uint32_t> num_buffers_per_channel = std::nullopt,
        bool reverse_order = false);

    // same as above but for vector of mesh
    static std::vector<ttnn::Tensor> invoke(
        const std::vector<ttnn::Tensor>& input_tensors,
        const std::optional<ttnn::Tensor>& persistent_output_buffer,  // TODO should this be a vector of tensors?
        int32_t dim,
        const std::vector<global_semaphore::MultiDeviceGlobalSemaphore>& multi_device_global_semaphore,
        uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
        std::optional<uint32_t> cluster_axis = std::nullopt,
        bool use_optimal_ccl_for_llama = false,
        const std::optional<std::vector<GlobalSemaphore>>& barrier_semaphore = std::nullopt,
        std::optional<uint32_t> chunks_per_sync = std::nullopt,
        std::optional<uint32_t> num_workers_per_link = std::nullopt,
        std::optional<uint32_t> num_buffers_per_channel = std::nullopt);

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
        bool use_optimal_ccl_for_llama = false,
        const std::optional<GlobalSemaphore>& barrier_semaphore = std::nullopt,
        bool reverse_order = false);
};

struct ExecuteAllGatherAsyncReversed {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        int32_t dim,
        const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
        uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
        bool use_optimal_ccl_for_llama = false,
        const std::optional<GlobalSemaphore>& barrier_semaphore = std::nullopt,
        bool reverse_order = false);

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
        bool use_optimal_ccl_for_llama = false,
        const std::optional<GlobalSemaphore>& barrier_semaphore = std::nullopt,
        std::optional<uint32_t> chunks_per_sync = std::nullopt,
        std::optional<uint32_t> num_workers_per_link = std::nullopt,
        std::optional<uint32_t> num_buffers_per_channel = std::nullopt,
        bool reverse_order = false);

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
        bool use_optimal_ccl_for_llama = false,
        const std::optional<GlobalSemaphore>& barrier_semaphore = std::nullopt,
        bool reverse_order = false);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto all_gather_async = ttnn::register_operation<
    "ttnn::experimental::all_gather_async",
    ttnn::operations::experimental::ccl::ExecuteAllGatherAsync>();

constexpr auto all_gather_async_reversed = ttnn::register_operation<
    "ttnn::experimental::all_gather_async_reversed",
    ttnn::operations::experimental::ccl::ExecuteAllGatherAsyncReversed>();

}  // namespace experimental
}  // namespace ttnn
