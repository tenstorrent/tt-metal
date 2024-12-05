// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

#include "ttnn/cpp/ttnn/operations/ccl/ccl_host_types.hpp"

namespace ttnn {
namespace operations {
namespace experimental {
namespace ccl {

struct ExecuteReduceScatter {
    static ttnn::Tensor invoke(
        const Tensor &input_tensor,
        const int32_t dim,
        const uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        ttnn::operations::reduction::ReduceType reduce_op = ttnn::operations::reduction::ReduceType::Sum,
        const std::optional<ttnn::MemoryConfig>& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear,
        const std::optional<size_t> num_links = std::nullopt);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const int32_t dim,
        ttnn::operations::reduction::ReduceType math_op,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear,
        const std::optional<size_t> num_links = std::nullopt);
};

}  // namespace ccl
} // namespace experimental
} // namespace operations

constexpr auto reduce_scatter_async =
    ttnn::register_operation<"ttnn::reduce_scatter_async", ttnn::operations::experimental::ccl::ExecuteReduceScatter>();

}  // namespace ttnn
