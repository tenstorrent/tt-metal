// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flip.hpp"
#include "ttnn/operations/core/core.hpp"
#include "cpp/ttnn/operations/data_movement/concat/concat.hpp"
#include "cpp/ttnn/operations/data_movement/chunk/chunk.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement {

ttnn::Tensor FlipOperation::invoke(const ttnn::Tensor& input_tensor, const std::vector<int>& dims) {
    auto size = input_tensor.get_logical_shape();
    int num_dims = size.rank();

    std::vector<int> flip_dims;
    for (int dim : dims) {
        int flip_dim = (dim < 0) ? (dim + num_dims) : dim;
        TT_FATAL(
            flip_dim >= 0 && flip_dim < num_dims,
            "Invalid dimension for flip operation, the dimensions should be in the range of [{}, {}]",
            -num_dims,
            num_dims - 1);
        flip_dims.push_back(flip_dim);
    }

    ttnn::Tensor output_tensor = input_tensor;
    for (int flip_dim : flip_dims) {
        int num_chunks = size[flip_dim];

        std::vector<ttnn::Tensor> chunks = ttnn::chunk(output_tensor, num_chunks, flip_dim);

        std::vector<ttnn::Tensor> reversed_chunks;
        reversed_chunks.reserve(num_chunks);
        for (int i = num_chunks - 1; i >= 0; --i) {
            reversed_chunks.push_back(chunks[i]);
        }

        output_tensor = ttnn::concat(reversed_chunks, flip_dim);
    }

    return output_tensor;
}

}  // namespace ttnn::operations::data_movement
