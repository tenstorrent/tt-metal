// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor plus_one(
    const Tensor& input_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    bool skip_negative_entries = false);

}  // namespace ttnn::operations::experimental
