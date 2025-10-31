// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "view.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::reshape {

ttnn::Tensor ViewOperation::invoke(
    const ttnn::Tensor& tensor, const ttnn::Shape& logical_shape, const ttnn::Shape& padded_shape) {
    return tt::tt_metal::operations::view(tensor, logical_shape, padded_shape);
}

ttnn::Tensor ViewOperation::invoke(const ttnn::Tensor& tensor, const ttnn::Shape& shape) {
    return tt::tt_metal::operations::view(tensor, shape, shape);
}

}  // namespace ttnn::operations::experimental::reshape
