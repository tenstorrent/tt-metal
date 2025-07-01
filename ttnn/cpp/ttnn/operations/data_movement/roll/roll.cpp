// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "roll.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"

namespace ttnn::operations::data_movement {

ttnn::Tensor RollOperation::invoke(
    const ttnn::Tensor& input_tensor, const ttnn::SmallVector<int>& shifts, const ttnn::SmallVector<int>& input_dims) {
    ttnn::Tensor result = input_tensor;
    auto size = result.logical_shape();
    int num_dims = size.rank();

    TT_FATAL(
        !shifts.empty() && shifts.size() == input_dims.size(),
        "Roll expects shifts {} and dims {} to have the same length",
        shifts.size(),
        input_dims.size());

    for (int dim : input_dims) {
        TT_FATAL(
            dim >= -num_dims && dim < num_dims,
            "Invalid dimension index {}. The dimension must be within the range [{}, {}].",
            dim,
            -num_dims,
            num_dims - 1);
    }

    std::vector<int> adjusted_shifts(shifts.begin(), shifts.end());

    for (size_t i = 0; i < adjusted_shifts.size(); ++i) {
        int shift = adjusted_shifts[i];
        int dim = input_dims[i];

        int shift_size = input_tensor.logical_shape()[dim];
        adjusted_shifts[i] = ((shift % shift_size) + shift_size) % shift_size;
    }

    const ttnn::SmallVector<int> stride_vector(num_dims, 1);

    for (size_t i = 0; i < adjusted_shifts.size(); ++i) {
        int dim = input_dims[i];

        if (dim < 0) {
            dim += num_dims;
        }

        int shift = adjusted_shifts[i] % size[dim];
        if (shift == 0) {
            continue;
        }

        ttnn::SmallVector<int> start_left(num_dims, 0), end_left;
        ttnn::SmallVector<int> start_right(num_dims, 0), end_right;

        for (int j = 0; j < num_dims; ++j) {
            end_left.push_back(size[j]);
            end_right.push_back(size[j]);
        }

        start_left[dim] = size[dim] - shift;
        start_right[dim] = 0;
        end_right[dim] = size[dim] - shift;

        ttnn::Tensor left_part = ttnn::slice(result, start_left, end_left, stride_vector);
        ttnn::Tensor right_part = ttnn::slice(result, start_right, end_right, stride_vector);

        std::vector<ttnn::Tensor> tensors_to_concat = {left_part, right_part};
        result = ttnn::concat(tensors_to_concat, dim);
    }

    return result;
}

ttnn::Tensor RollOperation::invoke(const ttnn::Tensor& input_tensor, const int shift) {
    ttnn::SmallVector<int> shifts = {shift};
    ttnn::SmallVector<int> dims = {1};  // Rolling will happen on dimension 1 after flattening

    auto original_shape = input_tensor.logical_shape();

    // Calculate total number of elements for flattening
    int total_elements = 1;
    for (int i = 0; i < original_shape.rank(); ++i) {
        total_elements *= original_shape[i];
    }

    // Flatten the input tensor to shape [1, total_elements]
    ttnn::Tensor result = ttnn::reshape(input_tensor, ttnn::Shape({1, total_elements}));

    result = invoke(result, shifts, dims);
    // Reshape back to the original shape
    result = ttnn::reshape(result, ttnn::Shape(original_shape));

    return result;
}

ttnn::Tensor RollOperation::invoke(const ttnn::Tensor& input_tensor, const int shift, const int dim) {
    ttnn::SmallVector<int> shifts = {shift};
    ttnn::SmallVector<int> dims = {dim};

    return invoke(input_tensor, shifts, dims);
}

}  // namespace ttnn::operations::data_movement
