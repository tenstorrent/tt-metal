// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unsqueeze.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::data_movement {

ttnn::Tensor UnsqueezeOperation::invoke(const ttnn::Tensor& input_tensor, const int dim) {
    const auto& tensor_shape = input_tensor.logical_shape();
    const uint32_t rank = tensor_shape.rank();
    const int32_t max_dim = (int)(rank);
    const int32_t min_dim = -(max_dim)-1;

    SmallVector<uint32_t> output_shape_vector;

    int normal_dim;
    // Handle negative dimension by converting it to positive
    TT_FATAL(
        (dim >= min_dim) && (dim <= max_dim),
        "Dimension out of range (expected to be in range of [{},{}], but got {})",
        min_dim,
        max_dim,
        dim);
    if (dim < 0) {
        normal_dim = rank + 1 + dim;
    } else {
        normal_dim = dim;
    }

    // Insert new dimension
    for (int i = 0; i < rank; ++i) {
        if (i == normal_dim) {
            output_shape_vector.push_back(1);
        }
        output_shape_vector.push_back(tensor_shape[i]);
    }

    // If the dimension is at the end, append it
    if (normal_dim == rank) {
        output_shape_vector.push_back(1);
    }

    return ttnn::reshape(input_tensor, ttnn::Shape(std::move(output_shape_vector)));
}

}  // namespace ttnn::operations::data_movement
