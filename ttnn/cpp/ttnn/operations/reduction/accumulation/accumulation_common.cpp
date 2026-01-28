// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "accumulation_common.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"

namespace ttnn::operations::reduction::accumulation::common {

Tensor preprocess_input_tensor(
    const Tensor& input_tensor,
    const int32_t& cum_axis,
    permutation_t& permutation,
    int32_t& accumulation_axis,
    std::optional<DataType>& dtype) {
    Tensor processed_tensor = input_tensor;
    const auto& input_dtype = input_tensor.dtype();
    if (dtype.has_value() && (input_dtype != *dtype)) {
        processed_tensor = ttnn::typecast(input_tensor, input_dtype, *dtype);
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
    }
    accumulation_axis = cum_axis;

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

void validate_output_tensor(const Tensor& input_tensor, const Tensor& output_tensor) {
    TT_FATAL(
        input_tensor.logical_shape() == output_tensor.logical_shape(),
        "Shape mismatch: input tensor shape {} does not match output tensor shape {}.",
        input_tensor.logical_shape(),
        output_tensor.logical_shape());
}

Tensor accumulation_invoke(
    const Tensor& input_tensor,
    int64_t dim,
    std::optional<ttnn::DataType> dtype,
    std::optional<Tensor> optional_out,
    const bool& reverse_order,
    const std::optional<MemoryConfig>& memory_config,
    ttnn::prim::AccumulationOp op) {
    const auto& input_shape = input_tensor.logical_shape();
    const int32_t& input_rank = input_shape.rank();

    if (input_rank == 0 || input_tensor.logical_volume() == 0) {
        return input_tensor;
    }

    TT_FATAL(
        ((dim >= -static_cast<decltype(dim)>(input_shape.rank())) &&
         (dim < static_cast<decltype(dim)>(input_shape.rank()))),
        "The requested accumulation axis is {}, while the input tensor has rank {}.",
        dim,
        input_tensor.padded_shape().rank());

    // Normalize negative dim
    const int32_t cum_axis = (dim < 0) ? (dim + input_rank) : dim;

    if (optional_out.has_value()) {
        validate_output_tensor(input_tensor, *optional_out);
    }

    Tensor wip_tensor = input_tensor;
    ttnn::SmallVector<int64_t> permutation;
    int32_t accumulation_axis;
    wip_tensor = common::preprocess_input_tensor(wip_tensor, cum_axis, permutation, accumulation_axis, dtype);
    wip_tensor = ttnn::prim::accumulation(
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
