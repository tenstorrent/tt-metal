// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

#include "ttnn/operations/ccl/ccl_host_types.hpp"

namespace ttnn {
namespace operations {
namespace ccl {

struct ExecuteReduceScatter {
    static ttnn::Tensor invoke(
        const Tensor& input_tensor,
        int32_t dim,
        uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        ttnn::operations::reduction::ReduceType reduce_op = ttnn::operations::reduction::ReduceType::Sum,
        uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& output_mem_config =
            tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
        std::optional<size_t> user_defined_num_workers = std::nullopt,
        std::optional<size_t> user_defined_num_buffers_per_channel = std::nullopt);

    static std::vector<ttnn::Tensor> invoke(
        const std::vector<ttnn::Tensor>& input_tensors,
        int32_t dim,
        uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        ttnn::operations::reduction::ReduceType reduce_op = ttnn::operations::reduction::ReduceType::Sum,
        uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& output_mem_config =
            tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
        std::optional<size_t> user_defined_num_workers = std::nullopt,
        std::optional<size_t> user_defined_num_buffers_per_channel = std::nullopt);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        int32_t dim,
        ttnn::operations::reduction::ReduceType math_op,
        uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
        std::optional<size_t> num_workers = std::nullopt,
        std::optional<size_t> num_buffers_per_channel = std::nullopt);

    static std::vector<ttnn::Tensor> invoke(
        const std::vector<ttnn::Tensor>& input_tensors,
        int32_t dim,
        ttnn::operations::reduction::ReduceType math_op,
        uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
        std::optional<size_t> num_workers = std::nullopt,
        std::optional<size_t> num_buffers_per_channel = std::nullopt);
};

}  // namespace ccl
}  // namespace operations

constexpr auto reduce_scatter =
    ttnn::register_operation<"ttnn::reduce_scatter", ttnn::operations::ccl::ExecuteReduceScatter>();

}  // namespace ttnn
