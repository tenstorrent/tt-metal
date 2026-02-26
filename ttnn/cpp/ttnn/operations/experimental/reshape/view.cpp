// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "view.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape_common.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::experimental {

ttnn::Tensor view(
    const ttnn::Tensor& tensor, const ttnn::Shape& logical_shape, const ttnn::Shape& padded_shape) {
    return tt::tt_metal::view(tensor, logical_shape, padded_shape);
}

ttnn::Tensor view(const ttnn::Tensor& tensor, const ttnn::Shape& shape) {
    return tt::tt_metal::view(tensor, shape, shape);
}

ttnn::Tensor view(const ttnn::Tensor& tensor, tt::stl::Span<const int32_t> shape_vector) {
    auto shape = ttnn::operations::data_movement::detail::infer_dims_for_reshape(tensor, shape_vector);
    return tt::tt_metal::view(tensor, shape, shape);
}

ttnn::Tensor view(const ttnn::Tensor& tensor, const ttnn::SmallVector<int32_t>& shape_vector) {
    return ttnn::experimental::view(tensor, tt::stl::Span<const int32_t>(shape_vector.data(), shape_vector.size()));
}

ttnn::Tensor view(const ttnn::Tensor& tensor, int32_t N, int32_t C, int32_t H, int32_t W) {
    ttnn::SmallVector<int32_t> shape_vec{N, C, H, W};
    return ttnn::experimental::view(tensor, tt::stl::Span<const int32_t>(shape_vec.data(), shape_vec.size()));
}

}  // namespace ttnn::experimental
