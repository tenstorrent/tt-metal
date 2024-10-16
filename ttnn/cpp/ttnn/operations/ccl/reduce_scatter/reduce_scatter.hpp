// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

#include "ttnn/cpp/ttnn/operations/ccl/ccl_host_types.hpp"

namespace ttnn {
namespace operations {
namespace ccl {

struct ExecuteReduceScatter {
    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor,
                               const uint32_t scatter_dim,
                               ttnn::operations::reduction::ReduceType math_op,
                               const uint32_t num_links = 1,
                               const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
                               ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
                               const std::optional<size_t> num_workers = std::nullopt,
                               const std::optional<size_t> num_buffers_per_channel = std::nullopt);
};

}  // namespace ccl
}  // namespace operations

constexpr auto reduce_scatter =
    ttnn::register_operation<"ttnn::reduce_scatter", ttnn::operations::ccl::ExecuteReduceScatter>();

}  // namespace ttnn
