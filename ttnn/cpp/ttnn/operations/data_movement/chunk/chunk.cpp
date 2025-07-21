// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "chunk.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement {

std::vector<ttnn::Tensor> ChunkOperation::invoke(const ttnn::Tensor& input_tensor, const uint32_t num_chunks, int dim) {
    TT_FATAL(num_chunks > 0, "Number of chunks must be greater than zero");

    auto size = input_tensor.logical_shape();
    int num_dims = size.rank();

    if (dim < 0) {
        dim += num_dims;
    }
    TT_FATAL(
        num_dims > dim, "... Invalid dimension for chunk operation, {} needs to be greater than {}", num_dims, dim);

    int size_along_dim = size[dim];
    int chunk_size = tt::div_up(size_along_dim, num_chunks);

    std::vector<ttnn::Tensor> chunks;
    int start = 0;

    while (start < size_along_dim) {
        int end = std::min(start + chunk_size, size_along_dim);

        ttnn::SmallVector<int> slice_start(num_dims, 0);
        ttnn::SmallVector<int> slice_end(num_dims);
        for (int i = 0; i < num_dims; ++i) {
            slice_end[i] = size[i];
        }
        slice_start[dim] = start;
        slice_end[dim] = end;
        ttnn::SmallVector<int> slice_step(num_dims, 1);

        ttnn::Tensor chunk_tensor = ttnn::slice(input_tensor, slice_start, slice_end, slice_step);

        chunks.push_back(chunk_tensor);

        start = end;
    }

    return chunks;
}

}  // namespace ttnn::operations::data_movement
