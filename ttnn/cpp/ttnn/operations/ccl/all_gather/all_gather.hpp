// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"

namespace ttnn {
namespace operations {
namespace ccl {

struct ExecuteAllGather {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        int32_t dim,
        uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        std::optional<size_t> num_workers = std::nullopt,
        std::optional<size_t> num_buffers_per_channel = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring);

    static std::vector<ttnn::Tensor> invoke(
        const std::vector<ttnn::Tensor>& input_tensors,
        int32_t dim,
        uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        std::optional<size_t> num_workers = std::nullopt,
        std::optional<size_t> num_buffers_per_channel = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        int32_t dim,
        uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        std::optional<size_t> num_workers = std::nullopt,
        std::optional<size_t> num_buffers_per_channel = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring);

    static std::vector<ttnn::Tensor> invoke(
        const std::vector<ttnn::Tensor>& input_tensors,
        int32_t dim,
        uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        std::optional<size_t> num_workers = std::nullopt,
        std::optional<size_t> num_buffers_per_channel = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring);
};

}  // namespace ccl
}  // namespace operations

constexpr auto all_gather = ttnn::register_operation<"ttnn::all_gather", ttnn::operations::ccl::ExecuteAllGather>();

}  // namespace ttnn
