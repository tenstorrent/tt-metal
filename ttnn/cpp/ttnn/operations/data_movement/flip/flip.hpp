// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/decorators.hpp"
#include <vector>

namespace ttnn {
namespace operations::data_movement {

struct FlipOperation {
    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor, const std::vector<int>& dims);
};

}  // namespace operations::data_movement

constexpr auto flip = ttnn::register_operation<"ttnn::flip", ttnn::operations::data_movement::FlipOperation>();

}  // namespace ttnn
