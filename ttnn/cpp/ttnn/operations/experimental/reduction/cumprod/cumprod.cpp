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

#include "tt-metalium/assert.hpp"
#include "cumprod.hpp"

namespace ttnn::operations::experimental::reduction {

const CumprodOperation::PermVec CumprodOperation::NATURAL_AXIS_ORDER{0, 1, 2, 3};

Tensor CumprodOperation::invoke(
    const Tensor& input_tensor,
    const int32_t dim,
    const std::optional<DataType>& input_dtype,
    const std::optional<Tensor>& optional_out,
    const std::optional<MemoryConfig>& memory_config,
    const QueueId& queue_id) {
    const auto output_memory_config = optional_out.has_value() ? optional_out.value().memory_config()
                                                               : memory_config.value_or(input_tensor.memory_config());
    const auto& padded_shape{input_tensor.get_padded_shape()};
    const auto& rank{padded_shape.rank()};
    if (rank == 0) {
        return input_tensor;  // TODO(jbbieniekTT): make sure about this
    }

    // get the cumulation axis as the normalized dim (abs trick)
    const auto cum_axis{(dim < 0) ? (rank - ((dim ^ (dim >> 31)) - (dim >> 31))) : dim};

    if (rank == FOUR_DIMENSIONS && cum_axis == FIXED_CUMULATION_AXIS) {
        // the input tensor and cum_axis are in the desired configuration
        return ttnn::prim::cumprod(input_tensor, dim, input_dtype, optional_out, output_memory_config, queue_id);
    } else {
        // permute (bring the input tensor to 4D and set cum_axis to FIXED_CUMULATION_AXIS)
        const auto [input_4d, permutation]{permute_to_4d(input_tensor, cum_axis)};
        // perform work on the preprocessed tensor
        const auto output_4d{
            ttnn::prim::cumprod(input_4d, dim, input_dtype, std::nullopt, input_4d.memory_config(), queue_id)};
        // bring back to the original shape + output to optional_out if provided
        return reorder_from_4d(output_4d, permutation, rank, optional_out);
    }
}

std::tuple<Tensor, CumprodOperation::PermVec> CumprodOperation::permute_to_4d(
    const Tensor& input_tensor, const uint32_t& cum_axis) {
    const auto& rank{input_tensor.get_padded_shape().rank()};
    TT_ASSERT(cum_axis != FIXED_CUMULATION_AXIS || rank != FOUR_DIMENSIONS);

    Tensor product{input_tensor};

    if (cum_axis == FIXED_CUMULATION_AXIS && rank == FOUR_DIMENSIONS) {
        // TODO(jbbieniekTT): TT_FATAL
    }

    if (rank < FOUR_DIMENSIONS) {
        // expand to 4D first
        for (uint32_t current_rank{rank}; current_rank < FOUR_DIMENSIONS; ++current_rank) {
            product = ttnn::unsqueeze(product, current_rank);
        }
    }

    PermVec permutation{NATURAL_AXIS_ORDER};
    if (cum_axis != FIXED_CUMULATION_AXIS) {
        // switch axes (cum_axis is set to the channel axis unconditionally)
        permutation[cum_axis] = FIXED_CUMULATION_AXIS;
        permutation[FIXED_CUMULATION_AXIS] = cum_axis;
        // perform permutation transmutation so that cum_axis goes along
        // the channel dimension (FIXED_CUMULATION_AXIS)
        product = ttnn::permute(product, permutation);
    }

    return {std::move(product), std::move(permutation)};
}

Tensor CumprodOperation::reorder_from_4d(
    const Tensor& input_tensor,
    const CumprodOperation::PermVec& permutation,
    const uint32_t& original_rank,
    const std::optional<Tensor>& optional_out) {
    TT_ASSERT(original_rank > 0 && original_rank <= FOUR_DIMENSIONS);
    const auto& rank{input_tensor.get_padded_shape().rank()};
    if (permutation == NATURAL_AXIS_ORDER && rank == FOUR_DIMENSIONS) {
        // TODO(jbbieniekTT): TT_FATAL
    }

    // the order of operations in permute_to_4d must be reverted here
    Tensor product{input_tensor};
    if (permutation != NATURAL_AXIS_ORDER) {
        // first reorder back since this is the last operation in preprocessing
        product = ttnn::permute(product, permutation);
    }
    if (original_rank < FOUR_DIMENSIONS) {
        // if the original rank isn't FOUR_DIMENSIONS, squeeze extraneous dimensions
        // (unsqueezing is the first step of preprocessing)
        for (uint32_t current_rank{rank}; current_rank > original_rank; --current_rank) {
            product = ttnn::squeeze(product, current_rank);
        }
    }
    if (optional_out.has_value()) {
        ttnn::copy(product, optional_out.value());
    }

    return product;
}

}  // namespace ttnn::operations::experimental::reduction
