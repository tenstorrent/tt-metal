// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::reduction::cumulation {

struct CumprodOperation {
    static Tensor invoke(
        const QueueId& queue_id,
        const Tensor& input_tensor,
        const int32_t& dim,
        std::optional<DataType>& dtype,
        std::optional<Tensor> optional_out,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace operations::reduction::cumulation

constexpr auto cumprod =
    ttnn::register_operation<"ttnn::cumprod", ttnn::operations::reduction::cumulation::CumprodOperation>();

}  // namespace ttnn
