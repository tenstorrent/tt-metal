// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "view.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape_common.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::reshape {

ttnn::Tensor ViewOperation::invoke(
    const ttnn::Tensor& tensor, const ttnn::Shape& logical_shape, const ttnn::Shape& padded_shape) {
    return tt::tt_metal::ops::view(tensor, logical_shape, padded_shape);
}

ttnn::Tensor ViewOperation::invoke(const ttnn::Tensor& tensor, const ttnn::Shape& shape) {
    return tt::tt_metal::ops::view(tensor, shape, shape);
}

ttnn::Tensor ViewOperation::invoke(const ttnn::Tensor& tensor, tt::stl::Span<const int32_t> shape_vector) {
    return invoke(tensor, ttnn::operations::data_movement::detail::infer_dims_for_reshape(tensor, shape_vector));
}

}  // namespace ttnn::operations::experimental::reshape
