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
        const uint32_t dim,
        const uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<size_t> num_workers = std::nullopt,
        const std::optional<size_t> num_buffers_per_channel = std::nullopt,
        const ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring);
};

}  // namespace ccl
}  // namespace operations

constexpr auto all_gather = ttnn::register_operation<"ttnn::all_gather", ttnn::operations::ccl::ExecuteAllGather>();

}  // namespace ttnn
