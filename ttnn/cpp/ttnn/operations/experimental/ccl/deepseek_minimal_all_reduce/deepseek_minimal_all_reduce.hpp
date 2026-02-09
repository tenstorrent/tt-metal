// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteDeepseekMinimalAllReduce {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        uint32_t num_links = 2,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear,
        std::optional<uint32_t> cluster_axis = std::nullopt,
        const std::optional<ttnn::Tensor>& intermediate_tensor = std::nullopt,
        const std::optional<ttnn::Tensor>& residual_tensor = std::nullopt,
        const std::optional<ttnn::Tensor>& persistent_output_tensor = std::nullopt);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto deepseek_minimal_all_reduce = ttnn::register_operation<
    "ttnn::experimental::deepseek_minimal_all_reduce",
    ttnn::operations::experimental::ccl::ExecuteDeepseekMinimalAllReduce>();

}  // namespace experimental
}  // namespace ttnn
