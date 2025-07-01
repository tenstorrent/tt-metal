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
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"

#include "tt-metalium/assert.hpp"
#include "cumprod.hpp"

namespace ttnn::operations::reduction {

Tensor CumprodOperation::invoke(
    const QueueId& queue_id,
    const Tensor& input_tensor,
    const int32_t& dim,
    std::optional<DataType>& dtype,
    std::optional<Tensor> optional_out,
    const std::optional<MemoryConfig>& memory_config) {
    const auto& input_shape = input_tensor.logical_shape();
    int tensor_rank = input_shape.rank();

    Tensor adjusted_input_tensor = input_tensor;
    const auto& input_dtype = input_tensor.dtype();

    constexpr uint32_t FOUR_DIMENSIONS{4};
    constexpr uint32_t FIRST_DIMENSION{0};

    if (tensor_rank == 0 || adjusted_input_tensor.logical_volume() == 0) {
        return adjusted_input_tensor;
    }

    // Normalize negative dim
    int32_t cum_axis{dim};
    if (cum_axis < 0) {
        cum_axis += tensor_rank;
    }

    // pre-/post-process WIP tensors if necessary to adjust kernel constraints
    if (tensor_rank - cum_axis < FOUR_DIMENSIONS) {
        int initial_tensor_rank = tensor_rank;
        if (initial_tensor_rank < FOUR_DIMENSIONS) {
            ttnn::SmallVector<uint32_t> new_dims = {};
            for (int i{initial_tensor_rank}; i < FOUR_DIMENSIONS; ++i) {
                new_dims.push_back(1);
            }
            new_dims.insert(new_dims.end(), input_shape.cbegin(), input_shape.cend());
            ttnn::Shape new_shape(new_dims);

            adjusted_input_tensor = ttnn::reshape(adjusted_input_tensor, new_shape);

            // Update params
            tensor_rank = FOUR_DIMENSIONS;
            cum_axis += (FOUR_DIMENSIONS - initial_tensor_rank);
        }

        // Create permutation that just swaps dim with the first dim
        ttnn::SmallVector<int64_t> permutation(tensor_rank);
        std::iota(permutation.begin(), permutation.end(), FIRST_DIMENSION);
        permutation[FIRST_DIMENSION] = cum_axis;
        permutation[cum_axis] = FIRST_DIMENSION;

        Tensor permuted_tensor =
            ttnn::permute(adjusted_input_tensor, permutation, adjusted_input_tensor.memory_config());

        // device cumprod works on the first dimension of 4
        Tensor output_tensor = ttnn::prim::cumprod(
            permuted_tensor,
            FIRST_DIMENSION,
            dtype,
            std::nullopt,
            memory_config.has_value() ? memory_config.value() : permuted_tensor.memory_config(),
            queue_id);

        // permute back
        output_tensor = ttnn::permute(output_tensor, permutation, output_tensor.memory_config());

        // reshape to the original form
        if (initial_tensor_rank < FOUR_DIMENSIONS) {
            output_tensor = ttnn::reshape(output_tensor, input_shape);
        }

        if (optional_out.has_value()) {
            ttnn::copy(output_tensor, *optional_out);
        }

        return output_tensor;
    }

    return ttnn::prim::cumprod(
        adjusted_input_tensor,
        cum_axis,
        dtype,
        optional_out,
        memory_config.has_value() ? memory_config.value() : adjusted_input_tensor.memory_config(),
        queue_id);
}

}  // namespace ttnn::operations::reduction
