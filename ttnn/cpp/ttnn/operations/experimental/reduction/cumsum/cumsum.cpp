// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/reduction/cumsum/cumsum.hpp"
#include <algorithm>
#include <iterator>
#include "tt-metalium/shape.hpp"
#include "tt-metalium/small_vector.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/experimental/reduction/cumsum/device/cumsum_device_operation.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::experimental::reduction {

Tensor CumSumOperation::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    int64_t dim,
    std::optional<ttnn::DataType> dtype,
    std::optional<Tensor> optional_output_tensor) {
    const auto& input_shape = input_tensor.get_logical_shape();
    int tensor_rank = input_shape.rank();

    Tensor adjusted_input_tensor = input_tensor;
    const auto& input_dtype = input_tensor.dtype();

    // TODO: Handle type conversion (convert input_tensor if necessary)
    // TODO: ttnn::to_dtype() does not seem to work with DeviceStorage (?)
    if (dtype.has_value() && input_dtype != dtype.value()) {
        // auto converted_tensor = ttnn::to_dtype(input_tensor, DataType::BFLOAT16);
        // adjusted_input_tensor = converted_tensor;

        // Create new tensor with proper dtype
        TensorSpec converted_specs = TensorSpec(
            input_tensor.get_logical_shape(),
            tt::tt_metal::TensorLayout(
                dtype.value(), tt::tt_metal::PageConfig(input_tensor.layout()), input_tensor.memory_config()));
        Tensor converted_tensor = tt::tt_metal::create_device_tensor(converted_specs, input_tensor.device());

        // Manually convert dtype
    }

    if (tensor_rank == 0 || adjusted_input_tensor.get_logical_volume() == 0) {  // empty input tensor => nothing to do
        return adjusted_input_tensor;
    }

    // Normalize negative dim
    if (dim < 0) {
        dim += tensor_rank;
    }

    // If dim is x or y axis (last or second last dimension)
    if (dim == tensor_rank - 1 || dim == tensor_rank - 2) {
        int initial_tensor_rank = tensor_rank;
        if (initial_tensor_rank <= 2) {  // 1D or 2D tensor
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

        // For now, the cumsum does not support `dim` == x or y-axis.
        // For now, we make the operation compatible by permuting axes if `dim` is either x or y axes.

        // Create permutation that just swaps dim with dim=0
        ttnn::SmallVector<int64_t> permutation(tensor_rank);
        std::iota(permutation.begin(), permutation.end(), 0);  // Initialize to [0,1,2,...]
        permutation[0] = dim;                                  // Swap dim with dim=0
        permutation[dim] = 0;

        Tensor permuted_tensor =
            ttnn::permute(adjusted_input_tensor, permutation, adjusted_input_tensor.memory_config());

        if (optional_output_tensor.has_value()) {
            optional_output_tensor =
                ttnn::permute(optional_output_tensor.value(), permutation, optional_output_tensor->memory_config());
        }

        // Compute cumsum on permuted tensor (now accumulation is on dim=0)
        Tensor output_tensor = ttnn::prim::cumsum(queue_id, permuted_tensor, 0, dtype, optional_output_tensor);

        // Apply backward permutation to restore initial shape
        output_tensor = ttnn::permute(output_tensor, permutation, output_tensor.memory_config());

        // if initial input tensor was 1D or 2D, then also reshape output to 1D or 2D
        if (initial_tensor_rank <= 2) {
            output_tensor = ttnn::reshape(output_tensor, input_shape);
        }

        return output_tensor;
    }

    // For other dimensions, proceed with original cumsum
    return ttnn::prim::cumsum(queue_id, adjusted_input_tensor, dim, dtype, optional_output_tensor);
}

}  // namespace ttnn::operations::experimental::reduction
