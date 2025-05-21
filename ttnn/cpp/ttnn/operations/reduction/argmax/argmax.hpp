// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {
namespace operations::reduction {

struct ArgMaxOperation {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const Tensor& input_tensor,
        const std::optional<int> dim = std::nullopt,
        const bool keepdim = false,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
        const bool use_muticore = false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt);
};

}  // namespace operations::reduction

constexpr auto argmax = ttnn::register_operation<"ttnn::argmax", ttnn::operations::reduction::ArgMaxOperation>();

}  // namespace ttnn
