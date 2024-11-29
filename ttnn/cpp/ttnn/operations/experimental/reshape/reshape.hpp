// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include <optional>

namespace ttnn {
namespace operations::experimental::reshape {


struct ReshapeOperation {
    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor, const ttnn::Shape& shape);
    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor, const ttnn::SimpleShape& shape);
};

}  // namespace operations::experimental::reshape

namespace experimental {
constexpr auto reshape =
    ttnn::register_operation_with_auto_launch_op<"ttnn::experimental::reshape", ttnn::operations::experimental::reshape::ReshapeOperation>();
}  // namespace experimental
}  // namespace ttnn
