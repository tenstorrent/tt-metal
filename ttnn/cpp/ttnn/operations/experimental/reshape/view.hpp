// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include <optional>

namespace ttnn::experimental {

ttnn::Tensor view(const ttnn::Tensor& input_tensor, const ttnn::Shape& shape);
ttnn::Tensor view(const ttnn::Tensor& input_tensor, tt::stl::Span<const int32_t> shape_vector);
ttnn::Tensor view(
    const ttnn::Tensor& input_tensor, const ttnn::Shape& logical_shape, const ttnn::Shape& padded_shape);

// Overloads for nanobind (SmallVector has a registered type caster; Span does not)
ttnn::Tensor view(const ttnn::Tensor& input_tensor, const ttnn::SmallVector<int32_t>& shape_vector);
ttnn::Tensor view(const ttnn::Tensor& input_tensor, int32_t N, int32_t C, int32_t H, int32_t W);

}  // namespace ttnn::experimental
