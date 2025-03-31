// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "squeeze.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::data_movement {

ttnn::Tensor SqueezeOperation::invoke(const ttnn::Tensor& input_tensor, const int dim) {
    const auto original_logical_shape = input_tensor.get_logical_shape();
    const auto padded_shape = input_tensor.get_padded_shape();
    const auto input_tensor_rank = original_logical_shape.rank();

    int normal_dim = dim;
    if (dim < 0) {
        // Handle negative dimension by converting it to positive
        normal_dim += input_tensor_rank;
    }

    // If dim is out of range or original dimension was not of size 1, include all dimensions
    if (normal_dim < 0 || normal_dim >= original_logical_shape.rank() || original_logical_shape[normal_dim] != 1) {
        return input_tensor;
    }

    SmallVector<uint32_t> original_logical_shape_vector(original_logical_shape.cbegin(), original_logical_shape.cend());
    SmallVector<uint32_t> padded_shape_vector(padded_shape.cbegin(), padded_shape.cend());
    original_logical_shape_vector.erase(original_logical_shape_vector.begin() + normal_dim);
    padded_shape_vector.erase(padded_shape_vector.begin() + normal_dim);

    return ttnn::reshape(
        input_tensor,
        ttnn::Shape(std::move(original_logical_shape_vector)),
        ttnn::Shape(std::move(padded_shape_vector)));
}

}  // namespace ttnn::operations::data_movement
