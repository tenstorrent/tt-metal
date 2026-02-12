// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {
namespace operations::experimental {

struct PlusOneOperation {
    static ttnn::Tensor invoke(
        const Tensor& input_tensor,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
        bool skip_negative_entries = false);
};

}  // namespace operations::experimental

constexpr auto plus_one =
    ttnn::register_operation<"ttnn::plus_one", ttnn::operations::experimental::PlusOneOperation>();

}  // namespace ttnn
