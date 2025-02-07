// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteSplitScatter {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const int32_t dim,
        const uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<size_t> num_workers = std::nullopt,
        const std::optional<size_t> num_buffers_per_channel = std::nullopt,
        const ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto split_scatter = ttnn::
    register_operation<"ttnn::experimental::split_scatter", ttnn::operations::experimental::ccl::ExecuteSplitScatter>();

}  // namespace experimental
}  // namespace ttnn
