// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "chunk.hpp"
#include "ttnn/operations/core/core.hpp"
#include "cpp/ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement {

std::vector<ttnn::Tensor> ChunkOperation::invoke(const ttnn::Tensor& input_tensor, const int num_chunks, int dim) {
    TT_FATAL(num_chunks > 0, "Number of chunks must be greater than zero");

    auto size = input_tensor.get_logical_shape();
    int num_dims = size.rank();

    if (dim < 0) {
        dim += num_dims;
    }
    TT_FATAL(num_dims > dim, "Invalid dimension for chunk operation");

    int size_along_dim = size[dim];
    int chunk_size = std::ceil(static_cast<float>(size_along_dim) / num_chunks);

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
        // std::cout << "Chunk " << chunks.size() << ": start[" << dim << "]=" << start << ", end[" << dim << "]=" <<
        // end << std::endl;

        ttnn::Tensor chunk_tensor = ttnn::Tensor(ttnn::slice(input_tensor, slice_start, slice_end, slice_step));

        chunks.push_back(chunk_tensor);

        start = end;
    }

    return chunks;
}

}  // namespace ttnn::operations::data_movement
