// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteDeepseekReduceScatter {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::MemoryConfig& output_memory_config,
        uint32_t num_links = 1,
        std::optional<uint32_t> cluster_axis = std::nullopt);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto deepseek_reduce_scatter = ttnn::register_operation<
    "ttnn::experimental::deepseek_reduce_scatter",
    ttnn::operations::experimental::ccl::ExecuteDeepseekReduceScatter>();

}  // namespace experimental
}  // namespace ttnn
