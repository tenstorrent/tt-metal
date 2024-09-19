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

    const auto tensor_shape = input_tensor.get_shape();
    const auto rank = tensor_shape.rank();
    std::vector<uint32_t> output_shape_vector;

    int normal_dim =  dim;
    if (dim < 0) {
        // Handle negative dimension by converting it to positive
        normal_dim += rank;
    }

    // Remove the dimension if it is of size 1
    for (size_t i = 0; i < tensor_shape.size(); ++i) {
        if (static_cast<int>(i) != normal_dim || tensor_shape[i] != 1) {
            output_shape_vector.push_back(tensor_shape[i]);
        }
    }

    // If dim is out of range or original dimension was not of size 1, include all dimensions
    if (dim >= static_cast<int>(tensor_shape.size()) || tensor_shape[dim] != 1) {
        return input_tensor;
    }

    ttnn::Shape output_shape(output_shape_vector);
    return ttnn::reshape(input_tensor, output_shape);

}

} // ttnn::operations::data_movement namespace
