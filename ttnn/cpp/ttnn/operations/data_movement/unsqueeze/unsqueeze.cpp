// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "unsqueeze.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::data_movement {

ttnn::Tensor UnsqueezeOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const int dim
    ) {

    const auto tensor_shape = input_tensor.get_shape();
    const auto rank = tensor_shape.rank();
    std::vector<uint32_t> output_shape_vector;

    TT_FATAL(input_tensor.get_layout() == Layout::ROW_MAJOR or (!tensor_shape.has_tile_padding()), "Currently supporing ROW-MAJOR tensors or TILE tensors with no padding");

    int normal_dim = dim;
    // Handle negative dimension by converting it to positive
    if (dim < 0) {
        normal_dim += rank + 1;
    }

    // Insert new dimension
    for (int i = 0; i < rank; ++i) {
        if (i == normal_dim) {
            output_shape_vector.push_back(1);
        }
        output_shape_vector.push_back(tensor_shape[i]);
    }

    // If the dimension is at the end, append it
    if (normal_dim >= tensor_shape.size()) {
        output_shape_vector.push_back(1);
    }

    return ttnn::reshape(input_tensor, ttnn::SimpleShape(std::move(output_shape_vector)));


}

} // ttnn::operations::data_movement namespace
