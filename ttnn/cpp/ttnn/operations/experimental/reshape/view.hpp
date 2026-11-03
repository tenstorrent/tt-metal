// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include <optional>

namespace ttnn::experimental {

ttnn::Tensor view(const ttnn::Tensor& input_tensor, const ttnn::Shape& shape);
ttnn::Tensor view(const ttnn::Tensor& input_tensor, ttsl::Span<const int32_t> shape_vector);
ttnn::Tensor view(
    const ttnn::Tensor& input_tensor, const ttnn::Shape& logical_shape, const ttnn::Shape& padded_shape);

// Overloads for nanobind (SmallVector has a registered type caster; Span does not)
ttnn::Tensor view(const ttnn::Tensor& input_tensor, const ttsl::SmallVector<int32_t>& shape_vector);
ttnn::Tensor view(const ttnn::Tensor& input_tensor, int32_t N, int32_t C, int32_t H, int32_t W);

// Zero-copy TILE view of an interleaved device tensor's buffer with a new shape and dtype. The view shares the
// input's memory and keeps it alive. The bytes are reinterpreted, not converted. Dim-0 slice i of the input owns one
// byte range on every bank, and the view's pages covering that range are a contiguous page range; see view.cpp.
ttnn::Tensor alias_view(const ttnn::Tensor& input_tensor, const ttnn::Shape& shape, tt::tt_metal::DataType dtype);

}  // namespace ttnn::experimental
