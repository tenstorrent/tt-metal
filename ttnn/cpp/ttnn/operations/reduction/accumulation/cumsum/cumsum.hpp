// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/common/queue_id.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::reduction::accumulation {

struct CumsumOperation {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& input,
        const int32_t& dim,
        std::optional<ttnn::DataType> dtype = std::nullopt,
        const bool& reverse_order = false,
        std::optional<Tensor> optional_out = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace ttnn::operations::reduction::accumulation

namespace ttnn {
constexpr auto cumsum =
    ttnn::register_operation<"ttnn::cumsum", ttnn::operations::reduction::accumulation::CumsumOperation>();

}  // namespace ttnn
