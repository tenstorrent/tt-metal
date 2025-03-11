// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::data_movement {

struct RollOperation {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor, const std::vector<int>& shifts, const std::vector<int>& dims);
};

}  // namespace operations::data_movement

constexpr auto roll = ttnn::register_operation<"ttnn::roll", ttnn::operations::data_movement::RollOperation>();

}  // namespace ttnn
