// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/global_semaphore.hpp"

#include <optional>

namespace ttnn::operations::experimental::ccl::transformer {

struct ExecuteAllReduceCreateQkvHeads {
    static std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> invoke(
        const ttnn::Tensor& input_tensor,
        ttnn::Tensor& buffer_tensor,
        const ttnn::Tensor& batch_offset,
        uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        const GlobalSemaphore& multi_device_global_semaphore,
        // create qkv heads non-optional parameters
        uint32_t num_heads,
        const std::optional<ttnn::MemoryConfig>& all_reduce_memory_config,
        ttnn::ccl::Topology topology,
        std::optional<size_t> num_preferred_links,
        std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt,
        // create qkv heads optional parameters
        std::optional<uint32_t> num_kv_heads,
        std::optional<uint32_t> slice_size = std::nullopt,
        const std::optional<MemoryConfig>& final_memory_config = std::nullopt,
        std::optional<DataType> dtype = std::nullopt,
        bool use_noc1_only = false);
};

}  // namespace ttnn::operations::experimental::ccl::transformer

namespace ttnn::experimental {

constexpr auto all_reduce_create_qkv_heads = ttnn::register_operation<
    "ttnn::experimental::all_reduce_create_qkv_heads",
    ttnn::operations::experimental::ccl::transformer::ExecuteAllReduceCreateQkvHeads>();

}  // namespace ttnn::experimental
