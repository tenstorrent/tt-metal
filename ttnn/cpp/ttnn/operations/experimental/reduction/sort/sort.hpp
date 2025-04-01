// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/decorators.hpp"

namespace ttnn::operations::experimental::reduction {

struct ExecuteSort {
    static std::vector<Tensor> invoke(
        QueueId queue_id,
        const Tensor& input_tensor,
        const int8_t dim,
        const bool descending,
        const bool stable,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<std::tuple<Tensor, Tensor>> optional_output_tensors = std::nullopt);

    static std::vector<Tensor> create_async_output_tensors(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs);
};

}  // namespace ttnn::operations::experimental::reduction

namespace ttnn::experimental {

constexpr auto sort = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::sort",
    ttnn::operations::experimental::reduction::ExecuteSort>();

}  // namespace ttnn::experimental
