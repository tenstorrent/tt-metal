// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations {
namespace mul_add {

struct MulAddOperation {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor_a, const ttnn::Tensor& input_tensor_b, const ttnn::Tensor& input_tensor_c);
};

}  // namespace mul_add
}  // namespace operations
constexpr auto mul_add =
    ttnn::register_operation_with_auto_launch_op<"ttnn::mul_add", ttnn::operations::mul_add::MulAddOperation>();

}  // namespace ttnn
