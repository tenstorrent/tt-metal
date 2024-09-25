// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "squeeze.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::data_movement {

ttnn::Tensor SqueezeOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const int dim
    ) {

    const auto original_logical_shape = input_tensor.get_shape();
    const auto padded_shape = input_tensor.get_shape().with_tile_padding();
    const auto input_tensor_rank = original_logical_shape.rank();

    int normal_dim =  dim;
    if (dim < 0) {
        // Handle negative dimension by converting it to positive
        normal_dim += input_tensor_rank;
    }

    std::vector<uint32_t> original_logical_shape_vector(input_tensor_rank - 1);
    std::vector<uint32_t> padded_shape_vector(input_tensor_rank - 1);
    uint32_t vector_id = 0;
    for(int i=0; i< input_tensor_rank; i++) {
        if(i != normal_dim or original_logical_shape[i] != 1) {
            original_logical_shape_vector[vector_id] = original_logical_shape[i];
            padded_shape_vector[vector_id] = padded_shape[i];
            vector_id++;
        }
    }

    // If dim is out of range or original dimension was not of size 1, include all dimensions
    if (normal_dim >= static_cast<int>(original_logical_shape.size()) || original_logical_shape[normal_dim] != 1) {
        return input_tensor;
    }

    return ttnn::reshape(input_tensor, ttnn::Shape(original_logical_shape_vector, padded_shape_vector));

}

} // ttnn::operations::data_movement namespace
