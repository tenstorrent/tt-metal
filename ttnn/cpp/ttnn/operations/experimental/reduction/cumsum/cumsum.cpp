// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/reduction/cumsum/cumsum.hpp"
#include <algorithm>
#include "tt-metalium/shape.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/experimental/reduction/cumsum/device/cumsum_device_operation.hpp"

namespace ttnn::operations::experimental::reduction {

Tensor CumSumOperation::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const int64_t dim,
    std::optional<ttnn::DataType> dtype,
    std::optional<Tensor> optional_output_tensor) {
    /*const auto& input_shape = input_tensor.get_logical_shape();
    int tensor_rank = input_shape.rank();

    std::vector<int> permutation;
    auto backward_permute = [](Tensor& a, const std::vector<int>& permutation){
        // Default = return a
        return a;
    };

    // If 1D 2D tensor => add 3rd dimension
    if (tensor_rank == 2) {
        std::vector<uint32_t> shape_vec;

        // TODO: Improve clarity
        if (tensor_rank == 1) {
            // [w] => [1, 1, w]
            shape_vec.push_back(1); // make 3D
            shape_vec.push_back(1);
        } else if (tensor_rank == 2) {
            // [h, w] => [1, h, w]
            shape_vec.push_back(1);
        }
        for (int i = 0; i < tensor_rank; i++) {
            shape_vec.push_back(input_shape[i]);
        }

        Shape new_shape(shape_vec);

        Tensor tensor_reshaped = ttnn::reshape_on_device(input_tensor, new_shape);
        tensor_rank = 3;

        // Assign correct tensor
        // TODO
    }

    // If dim is x or y axis
    // TODO: Handle negative `dim`
    if (dim + 1 == tensor_rank || dim + 2 == tensor_rank) {

        std::vector<uint32_t> shape_vec;
        for (int i = 0; i < tensor_rank; i++) {
            shape_vec.push_back(input_shape[i]);
        }

        std::vector<int> permutation(tensor_rank);
        std::iota(permutation.begin(), permutation.end(), 0);
        std::swap(permutation[dim], permutation[0]);

        // Forward permute
        Tensor reshaped_tensor = ttnn::permute(input_tensor, permutation, input_tensor.memory_config());

        backward_permute = [](Tensor& output_tensor, const std::vector<int>& permutation) {
            return ttnn::permute(output_tensor, permutation, output_tensor.memory_config());
        };

    }

    Tensor output_tensor = ttnn::prim::cumsum(queue_id, input_tensor, dim, dtype, optional_output_tensor);

    // TODO: Implement backward reshape
    backward_permute(output_tensor, permutation);

    return output_tensor;*/
    return ttnn::prim::cumsum(queue_id, input_tensor, dim, dtype, optional_output_tensor);
}

}  // namespace ttnn::operations::experimental::reduction
