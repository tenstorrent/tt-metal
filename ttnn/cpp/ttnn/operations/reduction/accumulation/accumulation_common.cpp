// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "accumulation_common.hpp"

namespace ttnn::operations::reduction::accumulation::common {

Tensor preprocess_input_tensor(
    const Tensor& input_tensor,
    const int32_t& cum_axis,
    permutation_t& permutation,
    int32_t& accumulation_axis,
    std::optional<DataType>& dtype) {
    Tensor processed_tensor = input_tensor;
    const auto& input_dtype = input_tensor.dtype();
    if (dtype.has_value() && (input_dtype != dtype.value())) {
        // auto converted_tensor = ttnn::to_dtype(input_tensor, DataType::BFLOAT16);
        // adjusted_input_tensor = converted_tensor;

        // Ideally, we would use `ttnn::to_dtype()` directly on input_tensor (DeviceStorage)
        // However, as of writing `ttnn::to_dtype()` does not support this.
        // The (provisional) workaround is to move the tensor to CPU, do the type conversion
        // and bring it back to the device.
        processed_tensor = processed_tensor.cpu();
        processed_tensor = ttnn::to_dtype(processed_tensor, dtype.value());

        processed_tensor = processed_tensor.to_device(input_tensor.device(), input_tensor.memory_config());
    }
    const auto& input_shape = processed_tensor.logical_shape();
    const auto& input_rank = input_shape.rank();
    if (input_rank - cum_axis < FOUR_DIMENSIONS) {
        int32_t final_rank = input_rank;
        int32_t final_cum_axis = cum_axis;
        if (input_rank < FOUR_DIMENSIONS) {
            ttnn::SmallVector<uint32_t> new_dims = {};
            for (int32_t i = input_rank; i < FOUR_DIMENSIONS; ++i) {
                new_dims.push_back(1);
            }
            new_dims.insert(new_dims.end(), input_shape.cbegin(), input_shape.cend());
            ttnn::Shape new_shape(new_dims);

            processed_tensor = ttnn::reshape(processed_tensor, new_shape);

            // Update params
            final_rank = FOUR_DIMENSIONS;
            final_cum_axis += (FOUR_DIMENSIONS - input_rank);
        }

        // Create permutation that just swaps cumulation axis with the first dim
        permutation = std::decay_t<decltype(permutation)>(final_rank);
        std::iota(permutation.begin(), permutation.end(), FIRST_DIMENSION);
        accumulation_axis = FIRST_DIMENSION;
        permutation[accumulation_axis] = final_cum_axis;
        permutation[final_cum_axis] = accumulation_axis;

        return ttnn::permute(processed_tensor, permutation, processed_tensor.memory_config());
    } else {
        accumulation_axis = cum_axis;
    }

    return processed_tensor;
}

Tensor postprocess_output_tensor(
    const Tensor& output_tensor,
    const int32_t& dim,
    const permutation_t& permutation,
    const ttnn::Shape& original_shape,
    const int32_t& original_rank) {
    Tensor processed_tensor = output_tensor;

    if (original_rank - dim < FOUR_DIMENSIONS) {
        processed_tensor = ttnn::permute(processed_tensor, permutation, processed_tensor.memory_config());
        if (original_rank < FOUR_DIMENSIONS) {
            processed_tensor = ttnn::reshape(processed_tensor, original_shape);
        }
    }

    return processed_tensor;
}

Tensor accumulation_invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    int64_t dim,
    std::optional<ttnn::DataType> dtype,
    std::optional<Tensor> optional_out,
    const bool& reverse_order,
    const std::optional<MemoryConfig>& memory_config,
    AccumulationOp op) {
    const auto& input_shape = input_tensor.logical_shape();
    const int32_t input_rank = input_shape.rank();

    const auto& input_dtype = input_tensor.dtype();

    if (input_rank == 0 || input_tensor.logical_volume() == 0) {
        return input_tensor;
    }

    // Normalize negative dim
    const int32_t cum_axis = (dim < 0) ? (dim + input_rank) : dim;

    Tensor wip_tensor = input_tensor;
    ttnn::SmallVector<int64_t> permutation;
    int32_t accumulation_axis;
    wip_tensor = common::preprocess_input_tensor(wip_tensor, cum_axis, permutation, accumulation_axis, dtype);
    wip_tensor = ttnn::prim::accumulation(
        queue_id,
        wip_tensor,
        accumulation_axis,
        dtype,
        reverse_order,
        std::nullopt,
        memory_config.has_value() ? memory_config.value() : wip_tensor.memory_config(),
        op);
    wip_tensor = common::postprocess_output_tensor(wip_tensor, cum_axis, permutation, input_shape, input_rank);
    if (optional_out.has_value()) {
        optional_out->storage() = wip_tensor.storage();
    }
    return wip_tensor;
}

}  // namespace ttnn::operations::reduction::accumulation::common
