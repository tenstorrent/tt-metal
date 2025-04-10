// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::experimental::reduction {

struct CumprodOperation {
    static Tensor invoke(
        const Tensor& input_tensor,
        const int32_t dim,
        std::optional<Tensor> optional_out = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const QueueId& queue_id = DefaultQueueId);
};

}  // namespace operations::experimental::reduction

namespace experimental {
constexpr auto cumprod = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::cumprod",
    ttnn::operations::experimental::reduction::CumprodOperation>();

}  // namespace experimental
}  // namespace ttnn
