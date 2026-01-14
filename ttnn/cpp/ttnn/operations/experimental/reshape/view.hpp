// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/decorators.hpp"
#include <optional>

namespace ttnn {
namespace operations::experimental::reshape {

struct ViewOperation {
    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor, const ttnn::Shape& shape);
    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor, tt::stl::Span<const int32_t> shape_vector);
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor, const ttnn::Shape& logical_shape, const ttnn::Shape& padded_shape);
};

}  // namespace operations::experimental::reshape

namespace experimental {
constexpr auto view =
    ttnn::register_operation<"ttnn::experimental::view", ttnn::operations::experimental::reshape::ViewOperation>();
}  // namespace experimental
}  // namespace ttnn
