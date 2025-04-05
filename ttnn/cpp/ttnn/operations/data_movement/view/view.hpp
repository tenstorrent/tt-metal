// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include <tt_stl/span.hpp>
#include "ttnn/decorators.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt {
namespace tt_metal {
class Shape;
}  // namespace tt_metal
}  // namespace tt

namespace ttnn {
namespace operations::data_movement {

struct ViewOperation {
    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor, const ttnn::Shape& logical_shape);
    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor, tt::stl::Span<const int32_t> shape_vector);
};

}  // namespace operations::data_movement
constexpr auto view = ttnn::register_operation<"ttnn::view", ttnn::operations::data_movement::ViewOperation>();
}  // namespace ttnn
