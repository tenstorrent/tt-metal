// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::data_movement {

struct StackOperation {
    static ttnn::Tensor invoke(const std::vector<ttnn::Tensor>& input_tensors, int dim);
};

}  // namespace operations::data_movement

constexpr auto stack = ttnn::register_operation<"ttnn::stack", ttnn::operations::data_movement::StackOperation>();

}  // namespace ttnn
