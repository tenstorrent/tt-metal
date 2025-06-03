// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/common/queue_id.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::reduction {

struct CumSumOperation {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& input,
        int64_t dim,
        std::optional<ttnn::DataType> dtype = std::nullopt,
        std::optional<Tensor> preallocated_output = std::nullopt,
        std::optional<bool> flip = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct CumSumBackwardOperation {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& input,
        int64_t dim,
        std::optional<ttnn::DataType> dtype = std::nullopt,
        std::optional<Tensor> preallocated_output = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace ttnn::operations::experimental::reduction

namespace ttnn::experimental {
constexpr auto cumsum = decorators::
    register_operation<"ttnn::experimental::cumsum", ttnn::operations::experimental::reduction::CumSumOperation>();

constexpr auto cumsum_backward = decorators::register_operation<
    "ttnn::experimental::cumsum_backward",
    ttnn::operations::experimental::reduction::CumSumBackwardOperation>();

}  // namespace ttnn::experimental
