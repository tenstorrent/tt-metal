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

const decltype(CumprodOperation::NATURAL_AXIS_ORDER) CumprodOperation::NATURAL_AXIS_ORDER{0, 1, 2, 3};
const decltype(CumprodOperation::PERMUTATIONS) CumprodOperation::PERMUTATIONS{
    // for dim == 0, swap batches and channels (batches go to FIXED_CUMULATION_AXIS)
    PermVec{1, 0, 2, 3},
    // for dim == 1 (FIXED_CUMULATION_AXIS or CHANNEL_DIMENSION) - no need to permute
    NATURAL_AXIS_ORDER,
    // for dim == 2, rotate tile dimensions with b/c dimensions + place dim == 2 onto FIXED_CUMULATION_AXIS
    PermVec{3, 2, 0, 1},
    // for dim == 3, rotate tile dimensions with b/c dimensions + place dim == 3 onto FIXED_CUMULATION_AXIS
    PermVec{2, 3, 0, 1}};

Tensor CumprodOperation::invoke(
    const Tensor& input_tensor,
    const int32_t& dim,
    std::optional<DataType>& dtype,
    std::optional<Tensor>& optional_out,
    const std::optional<MemoryConfig>& memory_config,
    const QueueId& queue_id) {
    // const auto output_memory_config = optional_out.has_value() ? optional_out.value().memory_config()
    //                                                            :
    //                                                            memory_config.value_or(input_tensor.memory_config());
    // const auto& padded_shape{input_tensor.get_padded_shape()};
    // const auto& rank{padded_shape.rank()};
    // if (rank == 0) {
    //     return input_tensor;  // TODO(jbbieniekTT): make sure about this
    // }

    // // get the cumulation axis as a normalized dim
    // const auto cum_axis{(dim < 0) ? (rank - abs(dim)) : dim};

    // if (rank == FOUR_DIMENSIONS && cum_axis == FIXED_CUMULATION_AXIS) {
    //     // the input tensor and cum_axis are in the desired configuration
    //     return ttnn::prim::cumprod(input_tensor, dim, input_dtype, optional_out, output_memory_config, queue_id);
    // } else {
    //     // bring the input tensor to the 4D form and set cum_axis to FIXED_CUMULATION_AXIS
    //     const auto [input_4d, permutation]{permute_to_4d(input_tensor, cum_axis)};
    //     // perform work on the preprocessed tensor
    //     const auto output_4d{
    //         ttnn::prim::cumprod(input_4d, dim, input_dtype, std::nullopt, input_4d.memory_config(), queue_id)};
    //     // bring back to the original shape + output to optional_out if provided
    //     return reorder_from_4d(output_4d, permutation, rank, optional_out);
    // }

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
        if (initial_tensor_rank <= 2) {
            // reshape tensor => make 3D or 4D
            ttnn::SmallVector<uint32_t> new_dims = {1, 1};
            new_dims.insert(new_dims.end(), input_shape.cbegin(), input_shape.cend());
            ttnn::Shape new_shape(new_dims);

            adjusted_input_tensor = ttnn::reshape(adjusted_input_tensor, new_shape);

            if (optional_out.has_value()) {
                optional_out = ttnn::reshape(optional_out.value(), new_shape);
            }

            tensor_rank += 2;
            cum_axis += 2;  // update dim parameter to target updated axis
        }

        // For now, cumprod does not support `dim` == x or y-axis.
        // We make it compatible by permuting axes if `dim` is either one of last two axes.

        // Create permutation that just swaps dim with dim=0
        ttnn::SmallVector<int64_t> permutation(tensor_rank);
        std::iota(permutation.begin(), permutation.end(), 0);  // Initialize to [0,1,2,...]
        permutation[0] = cum_axis;                             // Swap dim with dim=0
        permutation[cum_axis] = 0;

        Tensor permuted_tensor =
            ttnn::permute(adjusted_input_tensor, permutation, adjusted_input_tensor.memory_config());

        if (optional_out.has_value()) {
            optional_out = ttnn::permute(optional_out.value(), permutation, optional_out->memory_config());
        }

        // Compute cumprod on permuted tensor (now accumulation is on dim=0)
        Tensor output_tensor = ttnn::prim::cumprod(
            permuted_tensor,
            0,
            dtype,
            optional_out,
            memory_config.has_value() ? memory_config.value() : permuted_tensor.memory_config(),
            queue_id);

        // Apply backward permutation to restore initial shape
        output_tensor = ttnn::permute(output_tensor, permutation, output_tensor.memory_config());
        // TODO(jbbieniekTT): what about the optional out? (trying to handle it right now, not sure if correctly)
        if (optional_out.has_value()) {
            optional_out = ttnn::permute(optional_out.value(), permutation, optional_out.value().memory_config());
        }

        // if initial input tensor was 1D or 2D, then also reshape output to 1D or 2D
        if (initial_tensor_rank <= 2) {
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

std::tuple<Tensor, CumprodOperation::PermVec> CumprodOperation::permute_to_4d(
    const Tensor& input_tensor, const uint32_t& cum_axis) {
    const auto& rank{input_tensor.get_padded_shape().rank()};
    TT_ASSERT(cum_axis != FIXED_CUMULATION_AXIS || rank != FOUR_DIMENSIONS);

    Tensor product{input_tensor};

    if (rank < FOUR_DIMENSIONS) {
        // expand to 4D first
        for (uint32_t current_rank{rank}; current_rank < FOUR_DIMENSIONS; ++current_rank) {
            product = ttnn::unsqueeze(product, current_rank);
        }
    }

    PermVec permutation{PERMUTATIONS[cum_axis]};
    if (cum_axis != FIXED_CUMULATION_AXIS) {
        // for cum_axis > CHANNEL_DIMENSION:
        // there is an implication that data inside tiles is consecutively dependent
        // and the tensor must be permuted in order to provide the capability to compute
        // the cumprod - a proper permutation can rotate data in such a way that last two
        // axes are replaced with two first ones. this means that the new tensor will have
        // its tiles constructed from independent data, and consecutive tiles will contain
        // consecutively dependent data.
        // for cum_axis < CHANNEL_DIMENSION:
        // this case is easier - since consecutive dependent data is placed on parallel tiles,
        // full tiles can be fetched with some stride, the data being dependent *between*, not *inside*
        // them. It's been agreed that, for the sake of cache locality, accumulation shall ALWAYS
        // go along the channel axis (batches are further from each other), and a simple
        // batch-channel permutation of axes is performed.
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
        for (uint32_t current_axis{rank - 1}; current_axis > original_rank - 1; --current_axis) {
            product = ttnn::squeeze(product, current_axis);
        }
    }
    if (optional_out.has_value()) {
        ttnn::copy(product, optional_out.value());
    }

    return product;
}

}  // namespace ttnn::operations::experimental::reduction
