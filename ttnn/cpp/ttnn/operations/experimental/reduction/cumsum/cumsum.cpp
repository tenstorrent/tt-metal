// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/reduction/cumsum/cumsum.hpp"
#include <algorithm>
#include <iterator>
#include "tt-metalium/shape.hpp"
#include <tt_stl/small_vector.hpp>
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
    const auto& input_shape = input_tensor.logical_shape();
    int tensor_rank = input_shape.rank();

    Tensor adjusted_input_tensor = input_tensor;  // Tensor copy, but simplifies code (temporary solution)
    const auto& input_dtype = input_tensor.dtype();

    if (dtype.has_value() && input_dtype != dtype.value()) {
        // auto converted_tensor = ttnn::to_dtype(input_tensor, DataType::BFLOAT16);
        // adjusted_input_tensor = converted_tensor;

        // Ideally, we would use `ttnn::to_dtype()` directly on input_tensor (DeviceStorage)
        // However, as of writing `ttnn::to_dtype()` does not support this.
        // The (provisional) workaround is to move the tensor to CPU, do the type conversion
        // and bring it back to the device.
        Tensor cpu_tensor = input_tensor.cpu();
        Tensor cpu_converted_tensor = ttnn::to_dtype(cpu_tensor, dtype.value());

        Tensor converted_tensor = cpu_converted_tensor.to_device(input_tensor.device(), input_tensor.memory_config());

        adjusted_input_tensor = converted_tensor;
    }

    if (tensor_rank == 0 || adjusted_input_tensor.logical_volume() == 0) {  // empty input tensor => nothing to do

        if (optional_output_tensor.has_value()) {
            auto& out_tensor = optional_output_tensor.value();
            out_tensor.tensor_attributes->get_storage() =
                optional_output_tensor.value().tensor_attributes->get_storage();
        }

        return adjusted_input_tensor;
    }

    // Normalize negative dim
    if (dim < 0) {
        dim += tensor_rank;
    }

    // If dim is x or y axis (last or second last dimension)
    if (dim == tensor_rank - 1 || dim == tensor_rank - 2) {
        auto opt_output = optional_output_tensor;

        int initial_tensor_rank = tensor_rank;
        if (initial_tensor_rank <= 2) {  // 1D or 2D tensor
            // reshape tensor => make 3D or 4D
            ttnn::SmallVector<uint32_t> new_dims = {1, 1};
            new_dims.insert(new_dims.end(), input_shape.cbegin(), input_shape.cend());
            ttnn::Shape new_shape(new_dims);

            adjusted_input_tensor = ttnn::reshape(adjusted_input_tensor, new_shape);

            if (opt_output.has_value()) {
                opt_output = ttnn::reshape(optional_output_tensor.value(), new_shape);
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

        if (opt_output.has_value()) {
            opt_output = ttnn::permute(opt_output.value(), permutation, opt_output->memory_config());
        }

        // Compute cumsum on permuted tensor (now accumulation is on dim=0)
        Tensor output_tensor = ttnn::prim::cumsum(queue_id, permuted_tensor, 0, dtype, opt_output);

        // Apply backward permutation to restore initial shape
        output_tensor = ttnn::permute(output_tensor, permutation, output_tensor.memory_config());

        // if initial input tensor was 1D or 2D, then also reshape output to 1D or 2D
        if (initial_tensor_rank <= 2) {
            output_tensor = ttnn::reshape(output_tensor, input_shape);
        }

        if (opt_output.has_value()) {
            auto& out_tensor = optional_output_tensor.value();
            out_tensor.tensor_attributes->get_storage() = output_tensor.tensor_attributes->get_storage();
        }

        return output_tensor;
    }

    // For other dimensions, proceed with original cumsum
    return ttnn::prim::cumsum(queue_id, adjusted_input_tensor, dim, dtype, optional_output_tensor);
}

}  // namespace ttnn::operations::experimental::reduction
