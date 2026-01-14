// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/topk/topk.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"
#include "ttnn/operations/reduction/reduction_common/reduction_common.hpp"
#include "ttnn/operations/reduction/topk/device/topk_device_operation.hpp"
#include "ttnn/operations/reduction/topk/device/topk_constants.hpp"

#include <cstdint>

namespace ttnn::operations::reduction::topk {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

/**
 * @brief Rounds up K value to nearest tile-aligned boundary
 *
 * @param k The requested number of top elements to find
 * @return The tile-aligned K value (rounded up to next multiple of 32)
 *
 * Example: K=50 -> returns 64 (2 tiles of 32 elements each)
 *          K=32 -> returns 32 (exactly 1 tile)
 *          K=33 -> returns 64 (2 tiles needed)
 */
uint32_t get_nearest_supported_k_value(const uint32_t k) {
    return tt::constants::TILE_WIDTH * tt::div_up(k, tt::constants::TILE_WIDTH);
}

/**
 * @brief Applies post-processing transformations after TopK execution
 *
 * After the hardware TopK operation executes, several transformations are needed
 * to restore the output tensors to the expected format:
 *
 * 1. Slice adjustment: If K was rounded up for tile alignment, slice to actual K
 * 2. Rank restoration: Convert from 4D back to original tensor rank
 * 3. Dimension reordering: Transpose back if target dimension wasn't last
 * 4. Shape correction: Final slicing to match exact logical output shape
 *
 * This function consolidates all these transformations in correct order to ensure
 * the output matches user expectations while handling hardware constraints.
 *
 * @param input_tensor Original input tensor (for memory config reference)
 * @param result Vector containing [values_tensor, indices_tensor] from TopK op
 * @param dim The dimension along which TopK was performed
 * @param is_dim_last_idx Whether the target dimension was already last
 * @param k The actual requested K value
 * @param adjusted_k The tile-aligned K value used by hardware
 * @param original_lshape The original logical shape before transformations
 * @param input_memory_config Memory configuration to use for operations
 * @param sub_core_grids Core grid configuration (unused currently)
 * @param indices_tensor Optional indices tensor (unused currently)
 * @return Vector of [transformed_values, transformed_indices] tensors
 */
std::vector<Tensor> post_topk_transform_tensor(
    const Tensor& input_tensor,
    std::vector<Tensor>& result,
    const int8_t dim,
    const bool is_dim_last_idx,
    const uint32_t k,
    const uint32_t adjusted_k,
    const Shape& original_lshape,
    const MemoryConfig& input_memory_config,
    const CoreRangeSet& /*sub_core_grids*/,
    const std::optional<Tensor>& /*indices_tensor*/ = std::nullopt) {
    const auto& input_shape = input_tensor.padded_shape();
    const auto orig_rank = input_shape.rank();

    // Calculate the expected final logical shape after all transformations
    Shape final_lshape = original_lshape;
    final_lshape[dim] = std::min(original_lshape[dim], k);

    // Slice adjustment for tile-aligned K values
    // Hardware requires K to be tile-aligned (multiples of 32), but user wants exact K
    // If we had to round up K for hardware, slice down to the requested K value
    if (adjusted_k != k) {
        auto output_shape = result[0].padded_shape();
        ttnn::SmallVector<uint32_t> step = {1, 1, 1, 1};
        ttnn::SmallVector<uint32_t> start_index = {0, 0, 0, 0};
        ttnn::SmallVector<uint32_t> end_index = {output_shape[0], output_shape[1], output_shape[2], k};

        // Slice both values and indices tensors to remove extra elements beyond requested K
        result[0] = ttnn::slice(result[0], start_index, end_index, step, input_memory_config);
        result[1] = ttnn::slice(result[1], start_index, end_index, step, input_memory_config);
    }

    // Rank restoration - convert from hardware-required 4D back to original rank
    if (orig_rank < 4) {
        // For tensors originally < 4D, squeeze out the extra dimensions that were added
        result[0] = ttnn::squeeze_from_4D(result[0], orig_rank);
        result[1] = ttnn::squeeze_from_4D(result[1], orig_rank);
    } else if (orig_rank > 4) {
        // For tensors originally > 4D, reshape back to original higher-dimensional structure
        ttnn::SmallVector<uint32_t> result_shape(input_shape.cbegin(), input_shape.cend());
        result_shape[result_shape.size() - 1] = k;  // Update last dimension to K
        result[0] = ttnn::reshape(result[0], ttnn::Shape{result_shape});
        result[1] = ttnn::reshape(result[1], ttnn::Shape{result_shape});
    }

    // Dimension reordering - restore original dimension order
    // If we transposed the target dimension to be last for hardware processing,
    // transpose it back to its original position
    if (!is_dim_last_idx) {
        result[0] = ttnn::transpose(result[0], dim, -1, input_tensor.memory_config());
        result[1] = ttnn::transpose(result[1], dim, -1, input_tensor.memory_config());
    }

    // Final shape correction - ensure exact logical shape match
    // After all rank and dimension transformations, the logical shape might still not
    // match exactly due to padding or other hardware-imposed constraints.
    // This final slice ensures the output has the exact expected logical shape.
    if (result[0].logical_shape() != final_lshape) {
        int rank = final_lshape.rank();

        // Build slice parameters to extract exactly the expected shape
        ttnn::SmallVector<uint32_t> step;
        ttnn::SmallVector<uint32_t> start_index;
        ttnn::SmallVector<uint32_t> end_index;

        for (int i = 0; i < rank; i++) {
            step.push_back(1);                     // No skipping, take every element
            start_index.push_back(0);              // Start from beginning in each dimension
            end_index.push_back(final_lshape[i]);  // End at desired dimension size
        }

        // Apply final corrective slice to both values and indices tensors
        result[0] = ttnn::slice(result[0], start_index, end_index, step, input_memory_config);
        result[1] = ttnn::slice(result[1], start_index, end_index, step, input_memory_config);
    }

    // Verify that all transformations resulted in the correct output shape
    TT_FATAL(
        result[0].logical_shape() == final_lshape,
        "Output tensor transformation failed to create correct output shape!");

    return result;
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

std::vector<Tensor> ExecuteTopK::invoke(
    const Tensor& input_tensor,
    const uint32_t k,
    const int8_t dim,
    const bool largest,
    const bool sorted,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<Tensor>& indices_tensor,
    const std::optional<std::tuple<Tensor, Tensor>>& preallocated_output_tensors) {
    // Store original shape for final output validation
    const ttnn::Shape& original_lshape = input_tensor.logical_shape();

    // Analyze input tensor properties to determine required transformations
    auto rank = input_tensor.padded_shape().rank();
    const bool is_dim_last_idx = (dim == -1 || dim == rank - 1);
    const bool is_rank_le_4d = rank <= 4;

    // Normalize negative dimension index and validate K parameter
    const auto adjusted_dim = dim < 0 ? dim + rank : dim;
    TT_FATAL(
        input_tensor.logical_shape()[adjusted_dim] >= k,
        "K cannot be larger than the dimension size! K={}, dimension size={}",
        k,
        input_tensor.logical_shape()[adjusted_dim]);

    // Set up memory and execution configurations with defaults if not provided
    const auto input_memory_config = memory_config.value_or(input_tensor.memory_config());
    const auto used_sub_core_grids = sub_core_grids.value_or(ttnn::CoreRangeSet(
        ttnn::CoreRange(ttnn::CoreCoord(0, 0), input_tensor.device()->compute_with_storage_grid_size())));

    // Hardware constraint: K must be tile-aligned (multiple of 32 elements)
    // Round up to nearest supported value for hardware execution
    const uint32_t adjusted_k = CMAKE_UNIQUE_NAMESPACE::get_nearest_supported_k_value(k);

    // Dimension reordering - move target dimension to last position
    Tensor transposed_tensor = reduction_common::perform_transpose(input_tensor, is_dim_last_idx, dim, -1);

    // Rank normalization - convert to 4D tensor format
    Tensor transformed_tensor = reduction_common::transform_to_4d_tensor(transposed_tensor, is_rank_le_4d);

    // Dimension size padding - ensure minimum dimension size for efficient processing
    auto padded_tensor = transformed_tensor;
    const auto current_dim_size = static_cast<int>(transformed_tensor.logical_shape()[-1]);
    const auto min_required_size = static_cast<int>(ttnn::prim::constants::min_dim_per_core);
    const auto pad_amount = std::max(min_required_size - current_dim_size, 0);

    // Choose padding value based on whether we want largest or smallest values
    const auto pad_val = largest ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();

    if (pad_amount > 0) {
        ttnn::SmallVector<std::array<uint32_t, 2>> padding = {{0, 0}, {0, 0}, {0, 0}, {0, pad_amount}};

        // Use multicore padding for BFLOAT16 tensors not in L1 memory for better performance
        const bool pad_multicore = transformed_tensor.dtype() == DataType::BFLOAT16 &&
                                   transformed_tensor.memory_config().buffer_type() != BufferType::L1;
        padded_tensor = ttnn::pad(transformed_tensor, padding, pad_val, pad_multicore);
    }

    // Fill any implicit tile padding with appropriate values
    padded_tensor = ttnn::fill_implicit_tile_padding(padded_tensor, pad_val);

    // Execute TopK operation
    auto [output_value_tensor, output_index_tensor] = ttnn::prim::topk(
        padded_tensor,
        adjusted_k,
        -1,
        largest,
        sorted,
        input_memory_config,
        used_sub_core_grids,
        indices_tensor,
        preallocated_output_tensors);

    // Package results into vector format expected by post-processing
    std::vector<Tensor> output_tensor_vec;
    output_tensor_vec.reserve(2);
    output_tensor_vec.push_back(std::move(output_value_tensor));
    output_tensor_vec.push_back(std::move(output_index_tensor));

    // Apply post-processing transformations to restore original format
    return CMAKE_UNIQUE_NAMESPACE::post_topk_transform_tensor(
        transposed_tensor,
        output_tensor_vec,
        dim,
        is_dim_last_idx,
        k,
        adjusted_k,
        original_lshape,
        input_memory_config,
        used_sub_core_grids,
        indices_tensor);
}

}  // namespace ttnn::operations::reduction::topk
