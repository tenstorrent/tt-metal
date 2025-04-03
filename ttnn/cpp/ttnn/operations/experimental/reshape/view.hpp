// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/decorators.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt {
namespace tt_metal {
class Shape;
}  // namespace tt_metal
}  // namespace tt

namespace ttnn {
namespace operations::experimental::reshape {

struct ViewOperation {
    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor, const ttnn::Shape& shape);
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor, const ttnn::Shape& logical_shape, const ttnn::Shape& padded_shape);
};

}  // namespace operations::experimental::reshape

namespace experimental {
constexpr auto view = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::view",
    ttnn::operations::experimental::reshape::ViewOperation>();
}  // namespace experimental
}  // namespace ttnn
