// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {
namespace operations::experimental {

struct PlusOneOperation {
    static ttnn::Tensor invoke(
        QueueId queue_id, const Tensor& input_tensor, const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
};

}  // namespace operations::experimental

constexpr auto plus_one =
    ttnn::register_operation<"ttnn::plus_one", ttnn::operations::experimental::PlusOneOperation>();

}  // namespace ttnn
