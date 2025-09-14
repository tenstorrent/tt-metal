// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::reduction::accumulation {

struct CumprodOperation {
    static Tensor invoke(
        const QueueId& queue_id,
        const Tensor& input_tensor,
        const int32_t& dim,
        std::optional<DataType>& dtype,
        const bool& reverse_order,
        std::optional<Tensor> optional_out,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace operations::reduction::accumulation

constexpr auto cumprod =
    ttnn::register_operation<"ttnn::cumprod", ttnn::operations::reduction::accumulation::CumprodOperation>();

}  // namespace ttnn
