// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "squeeze.hpp"
#include <tt_stl/small_vector.hpp>
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::data_movement {

ttnn::Tensor SqueezeOperation::invoke(const ttnn::Tensor& input_tensor, const ttnn::SmallVector<int>& dim) {
    const auto& original_logical_shape = input_tensor.logical_shape();
    const auto& padded_shape = input_tensor.padded_shape();
    auto input_tensor_rank = original_logical_shape.rank();

    SmallVector<uint32_t> new_logical_shape(original_logical_shape.cbegin(), original_logical_shape.cend());
    SmallVector<uint32_t> new_padded_shape(padded_shape.cbegin(), padded_shape.cend());

    // Explicitly copy dim to avoid modifying the input
    auto dims = dim;

    // handle negative dimensions
    for (size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] < 0) {
            dims[i] += input_tensor_rank;
        }
    }
    // Sort the dimensions in descending order to avoid issues with modifying new_shape in loop
    std::sort(dims.rbegin(), dims.rend());

    // Special ugly case for 0-ranked input
    if (input_tensor_rank == 0) [[unlikely]] {
        if (dims.empty() || (dims.size() == 1 && (dims[0] == 0 || dims[0] == -1))) {
            return input_tensor;
        }
        TT_THROW("Dimension out of range (expected to be of [-1, 0], but got {})", dims[0]);
    }

    for (size_t i = 0; i < dims.size(); ++i) {
        const auto dim = dims[i];
        // Check duplicate dimensions
        if (i > 0) {
            TT_FATAL(dim != dims[i - 1], "dim {} appears multiple times in the list of dims", dim);
        }
        TT_FATAL(
            (dim >= 0) && (dim < input_tensor_rank),
            "Dimension out of range (expected to be in range of [{},{}], but got {})",
            -static_cast<std::ptrdiff_t>(input_tensor_rank),
            input_tensor_rank - 1,
            dim);

        // If original dimension was not of size 1, include all dimensions
        if (original_logical_shape[dim] != 1) {
            continue;
        }

        new_logical_shape.erase(new_logical_shape.begin() + dim);
        new_padded_shape.erase(new_padded_shape.begin() + dim);
    }

    // Note: don't have to check padded too
    if (new_logical_shape == original_logical_shape) {
        return input_tensor;
    }

    return ttnn::reshape(
        input_tensor, ttnn::Shape(std::move(new_logical_shape)), ttnn::Shape(std::move(new_padded_shape)));
}

ttnn::Tensor SqueezeOperation::invoke(const ttnn::Tensor& input_tensor, int dim) {
    ttnn::SmallVector<int> dims{dim};
    return invoke(input_tensor, dims);
}

ttnn::Tensor SqueezeOperation::invoke(const ttnn::Tensor& input_tensor) {
    auto input_tensor_rank = input_tensor.logical_shape().rank();
    ttnn::SmallVector<int> dims(input_tensor_rank);
    std::iota(dims.begin(), dims.end(), 0);
    return invoke(input_tensor, dims);
}

}  // namespace ttnn::operations::data_movement
