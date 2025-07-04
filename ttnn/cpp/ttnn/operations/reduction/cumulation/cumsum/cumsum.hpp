// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/common/queue_id.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::reduction::cumulation {

struct CumsumOperation {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& input,
        int64_t dim,
        std::optional<ttnn::DataType> dtype = std::nullopt,
        std::optional<Tensor> preallocated_output = std::nullopt,
        const bool& flip = false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct CumsumBackwardOperation {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& input,
        int64_t dim,
        std::optional<ttnn::DataType> dtype = std::nullopt,
        std::optional<Tensor> preallocated_output = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace ttnn::operations::reduction::cumulation

namespace ttnn {
constexpr auto cumsum =
    decorators::register_operation<"ttnn::cumsum", ttnn::operations::reduction::cumulation::CumsumOperation>();

constexpr auto cumsum_backward = decorators::
    register_operation<"ttnn::cumsum_backward", ttnn::operations::reduction::cumulation::CumsumBackwardOperation>();

}  // namespace ttnn
