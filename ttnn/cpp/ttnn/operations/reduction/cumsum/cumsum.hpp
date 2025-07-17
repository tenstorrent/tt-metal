// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/common/queue_id.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::reduction {

struct CumSumOperation {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& input,
        int64_t dim,
        std::optional<ttnn::DataType> dtype = std::nullopt,
        std::optional<Tensor> preallocated_output = std::nullopt,
        bool flip = false,
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

}  // namespace ttnn::operations::reduction

namespace ttnn {
constexpr auto cumsum = decorators::register_operation<"ttnn::cumsum", ttnn::operations::reduction::CumSumOperation>();

constexpr auto cumsum_backward =
    decorators::register_operation<"ttnn::cumsum_backward", ttnn::operations::reduction::CumSumBackwardOperation>();

}  // namespace ttnn
