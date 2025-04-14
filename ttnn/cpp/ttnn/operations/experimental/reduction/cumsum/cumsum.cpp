// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/reduction/cumsum/cumsum.hpp"
#include <algorithm>
#include <iterator>
#include "tt-metalium/shape.hpp"
#include "tt-metalium/small_vector.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/experimental/reduction/cumsum/device/cumsum_device_operation.hpp"

namespace ttnn::operations::experimental::reduction {

Tensor CumSumOperation::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    int64_t dim,
    std::optional<ttnn::DataType> dtype,
    std::optional<Tensor> optional_output_tensor) {
    const auto& input_shape = input_tensor.get_logical_shape();
    int tensor_rank = input_shape.rank();

    // TODO: Handle type conversion (convert input_tensor if necessary)
    // TODO: Make sure we don't accidentaly modify input

    Tensor adjusted_input_tensor = input_tensor;
    const auto& input_dtype = input_tensor.dtype();
    // TODO: Create copy of input with dtype
    // if (dtype.has_value() && input_dtype != dtype.value()) {
    //    adjusted_input_tensor = ttnn::operations::core::detail::convert_to_dtype(input_tensor,
    //    input_tensor.get_layout(), dtype.value());
    //}

    if (tensor_rank == 0 || adjusted_input_tensor.get_logical_volume() == 0) {  // empty input tensor => nothing to do
        return adjusted_input_tensor;
    }

    // Normalize negative dim
    if (dim < 0) {
        dim += tensor_rank;
    }

    // If dim is x or y axis (last or second last dimension)
    if (dim == tensor_rank - 1 || dim == tensor_rank - 2) {
        // NOTE: Handle reshaping/permuting optional output tensor

        int initial_tensor_rank = tensor_rank;
        if (initial_tensor_rank <= 2) {
            // reshape tensor => make 3D or 4D
            ttnn::SmallVector<uint32_t> new_dims = {1, 1};
            new_dims.insert(new_dims.end(), input_shape.cbegin(), input_shape.cend());
            ttnn::Shape new_shape(new_dims);

            adjusted_input_tensor = ttnn::reshape(adjusted_input_tensor, new_shape);

            if (optional_output_tensor.has_value()) {
                optional_output_tensor = ttnn::reshape(optional_output_tensor.value(), new_shape);
            }

            tensor_rank += 2;
            dim += 2;  // update dim parameter to target updated axis
        }

        // Create permutation that just swaps dim with dim=0
        ttnn::SmallVector<int64_t> permutation(tensor_rank);
        std::iota(permutation.begin(), permutation.end(), 0);  // Initialize to [0,1,2,...]
        permutation[0] = dim;                                  // Swap dim with dim=0
        permutation[dim] = 0;

        // Permute dimensions
        Tensor permuted_tensor =
            ttnn::permute(adjusted_input_tensor, permutation, adjusted_input_tensor.memory_config());

        if (optional_output_tensor.has_value()) {
            optional_output_tensor =
                ttnn::permute(optional_output_tensor.value(), permutation, optional_output_tensor->memory_config());
        }

        // Compute cumsum on permuted tensor (now accumulation is on dim=0)
        Tensor output_tensor = ttnn::prim::cumsum(queue_id, permuted_tensor, 0, dtype, optional_output_tensor);

        // Backward permute to restore original dimension order (same permutation works in reverse)
        output_tensor = ttnn::permute(output_tensor, permutation, output_tensor.memory_config());

        // if necessary, reshape output to match input shape
        if (initial_tensor_rank <= 2) {
            output_tensor = ttnn::reshape(output_tensor, input_shape);
        }

        return output_tensor;
    }

    // For other dimensions, proceed with original cumsum
    return ttnn::prim::cumsum(queue_id, adjusted_input_tensor, dim, dtype, optional_output_tensor);
}

}  // namespace ttnn::operations::experimental::reduction
