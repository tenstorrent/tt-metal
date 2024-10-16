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

struct ExecuteAllReduce {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const uint32_t scatter_dim,
        ttnn::operations::reduction::ReduceType math_op,
        const uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
        const std::optional<size_t> num_workers = std::nullopt,
        const std::optional<size_t> num_buffers_per_channel = std::nullopt);
};

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

namespace experimental {
constexpr auto all_reduce =
    ttnn::register_operation<"ttnn::experimental::all_reduce", ttnn::operations::experimental::ccl::ExecuteAllReduce>();

}  // namespace experimental

}  // namespace ttnn
