// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/cumprod_device_operation.hpp"

#include <tuple>

#include <magic_enum/magic_enum.hpp>

#include <ttnn/operations/data_movement/copy/copy.hpp>
#include <ttnn/operations/data_movement/permute/permute.hpp>
#include <ttnn/operations/data_movement/squeeze/squeeze.hpp>
#include <ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp>
#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"

#include "tt-metalium/assert.hpp"
#include "cumprod.hpp"

namespace ttnn::operations::experimental::reduction {

Tensor CumprodOperation::invoke(
    const Tensor& input_tensor,
    const int32_t& dim,
    std::optional<DataType>& dtype,
    std::optional<Tensor>& optional_out,
    const std::optional<MemoryConfig>& memory_config,
    const QueueId& queue_id) {
    const auto& input_shape = input_tensor.get_logical_shape();
    int tensor_rank = input_shape.rank();

    Tensor adjusted_input_tensor = input_tensor;
    const auto& input_dtype = input_tensor.dtype();

    // TODO(jbbieniekTT): enable once ttnn::to_dtype works for tensors on device storage
    // if (dtype.has_value() && input_dtype != dtype.value()) {
    //    auto converted_tensor = ttnn::to_dtype(input_tensor, DataType::BFLOAT16);
    //    adjusted_input_tensor = converted_tensor;
    // }

    if (tensor_rank == 0 || adjusted_input_tensor.get_logical_volume() == 0) {  // empty input tensor => nothing to do
        return adjusted_input_tensor;
    }

    // Normalize negative dim
    uint32_t cum_axis{dim};
    if (cum_axis < 0) {
        cum_axis += tensor_rank;
    }

    // If dim is either one of two last dimensions
    if (cum_axis == tensor_rank - 1 || cum_axis == tensor_rank - 2) {
        int initial_tensor_rank = tensor_rank;
        if (initial_tensor_rank < 4) {
            ttnn::SmallVector<uint32_t> new_dims = {};
            for (int i{initial_tensor_rank}; i < 4; ++i) {
                new_dims.push_back(1);
            }
            new_dims.insert(new_dims.end(), input_shape.cbegin(), input_shape.cend());
            ttnn::Shape new_shape(new_dims);

            adjusted_input_tensor = ttnn::reshape(adjusted_input_tensor, new_shape);

            if (optional_out.has_value()) {
                optional_out = ttnn::reshape(optional_out.value(), new_shape);
            }

            tensor_rank += (4 - initial_tensor_rank);
            cum_axis += (4 - initial_tensor_rank);  // update dim parameter to target updated axis
        }

        // Create permutation that just swaps dim with the first dim
        ttnn::SmallVector<int64_t> permutation(tensor_rank);
        std::iota(permutation.begin(), permutation.end(), 0);
        permutation[0] = cum_axis;
        permutation[cum_axis] = 0;

        Tensor permuted_tensor =
            ttnn::permute(adjusted_input_tensor, permutation, adjusted_input_tensor.memory_config());

        if (optional_out.has_value()) {
            optional_out = ttnn::permute(optional_out.value(), permutation, optional_out->memory_config());
        }

        Tensor output_tensor = ttnn::prim::cumprod(
            permuted_tensor,
            0,
            dtype,
            optional_out,
            memory_config.has_value() ? memory_config.value() : permuted_tensor.memory_config(),
            queue_id);

        output_tensor = ttnn::permute(output_tensor, permutation, output_tensor.memory_config());
        // TODO(jbbieniekTT): what about the optional out? (trying to handle it right now, not sure if correctly)
        if (optional_out.has_value()) {
            optional_out = ttnn::permute(optional_out.value(), permutation, optional_out.value().memory_config());
        }

        if (initial_tensor_rank < 4) {
            output_tensor = ttnn::reshape(output_tensor, input_shape);
            if (optional_out.has_value()) {
                optional_out = ttnn::reshape(optional_out.value(), input_shape);
            }
        }

        return output_tensor;
    }

    // For other dimensions, proceed with original cumprod
    return ttnn::prim::cumprod(
        adjusted_input_tensor,
        cum_axis,
        dtype,
        optional_out,
        memory_config.has_value() ? memory_config.value() : adjusted_input_tensor.memory_config(),
        queue_id);
}

}  // namespace ttnn::operations::experimental::reduction
