// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteLlamaReduceScatter {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const int32_t dim,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace operations::experimental::ccl

namespace experimental {
constexpr auto llama_reduce_scatter = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::llama_reduce_scatter",
    ttnn::operations::experimental::ccl::ExecuteLlamaReduceScatter>();
}  // namespace experimental

}  // namespace ttnn
