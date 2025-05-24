// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include <vector>

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::reduction {

struct ExecuteTopK {
    static std::vector<Tensor> invoke(
        QueueId queue_id,
        const Tensor& input_tensor,
        const uint32_t k,
        const int8_t dim,
        const bool largest,
        const bool sorted,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<CoreRangeSet>& sub_core_grids,
        std::optional<std::tuple<Tensor, Tensor>> optional_output_tensors = std::nullopt);
};

}  // namespace operations::reduction

constexpr auto topk = ttnn::register_operation<"ttnn::topk", ttnn::operations::reduction::ExecuteTopK>();

}  // namespace ttnn
