// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "roll.hpp"
#include "ttnn/operations/core/core.hpp"
#include "cpp/ttnn/operations/data_movement/slice/slice.hpp"
#include "cpp/ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement {

ttnn::Tensor RollOperation::invoke(
    const ttnn::Tensor& input_tensor, const std::vector<int>& shifts, const std::vector<int>& dims) {
    TT_FATAL(
        !shifts.empty() && shifts.size() == dims.size(),
        "Roll expects shifts {} and dims {} to have the same length",
        shifts.size(),
        dims.size());

    std::vector<int> adjusted_shifts = shifts;

    for (size_t i = 0; i < adjusted_shifts.size(); ++i) {
        int shift = adjusted_shifts[i];
        int dim = dims[i];

        int shift_size = input_tensor.get_logical_shape()[dim];
        adjusted_shifts[i] = ((shift % shift_size) + shift_size) % shift_size;
    }

    ttnn::Tensor result = input_tensor;
    auto size = result.get_logical_shape();
    int num_dims = size.rank();

    for (int dim : dims) {
        TT_FATAL(
            dim >= -num_dims && dim < num_dims,
            "Invalid dimension index {}. The dimension must be within the range [{}, {}].",
            dim,
            -num_dims,
            num_dims - 1);
    }

    for (size_t i = 0; i < adjusted_shifts.size(); ++i) {
        int dim = dims[i];

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

        ttnn::Tensor left_part = ttnn::slice(result, start_left, end_left, ttnn::SmallVector<int>(num_dims, 1));
        ttnn::Tensor right_part = ttnn::slice(result, start_right, end_right, ttnn::SmallVector<int>(num_dims, 1));

        std::vector<ttnn::Tensor> tensors_to_concat = {left_part, right_part};
        result = ttnn::concat(tensors_to_concat, dim);
    }

    return result;
}

}  // namespace ttnn::operations::data_movement
