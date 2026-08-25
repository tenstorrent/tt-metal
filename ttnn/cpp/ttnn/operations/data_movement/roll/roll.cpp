// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "roll.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/core/to_memory_config/to_memory_config_op.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/tilize/tilize.hpp"
#include "ttnn/operations/data_movement/untilize/untilize.hpp"
#include "ttnn/operations/data_movement/roll/device/roll_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"

namespace ttnn {

ttnn::Tensor roll(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<int>& shifts,
    const ttsl::SmallVector<int>& input_dims,
    const std::optional<MemoryConfig>& memory_config) {
    ttnn::Tensor result = input_tensor;
    auto size = result.logical_shape();
    int num_dims = size.rank();

    TT_FATAL(
        !shifts.empty() && shifts.size() == input_dims.size(),
        "Roll expects shifts {} and dims {} to have the same length",
        shifts.size(),
        input_dims.size());

    for (int dim : input_dims) {
        TT_FATAL(
            dim >= -num_dims && dim < num_dims,
            "Invalid dimension index {}. The dimension must be within the range [{}, {}].",
            dim,
            -num_dims,
            num_dims - 1);
    }

    std::vector<int> adjusted_shifts(shifts.begin(), shifts.end());

    for (size_t i = 0; i < adjusted_shifts.size(); ++i) {
        int shift = adjusted_shifts[i];
        int dim = input_dims[i];

        int shift_size = input_tensor.logical_shape()[dim];
        adjusted_shifts[i] = ((shift % shift_size) + shift_size) % shift_size;
    }

    const ttsl::SmallVector<int> stride_vector(num_dims, 1);

    // Sharded inputs use the native sharded roll device op, applied one dim at a time. A
    // tilized roll is native only when shifts on the last two dims are tile-aligned (a
    // whole-tile permutation); otherwise untilize/roll/tilize while staying sharded.
    const bool is_sharded = input_tensor.is_sharded();
    const bool is_tile = input_tensor.layout() == ttnn::TILE_LAYOUT;
    // Preserve input layout by default; caller may override with an explicit memory_config.
    const auto& native_mem_config = input_tensor.memory_config();
    const auto output_mem_config = memory_config.value_or(native_mem_config);

    if (is_sharded) {
        bool native_ok = true;
        if (is_tile) {
            constexpr int tile_dim = 32;
            for (size_t i = 0; i < adjusted_shifts.size(); ++i) {
                int dim = input_dims[i];
                if (dim < 0) {
                    dim += num_dims;
                }
                if ((dim == num_dims - 1 || dim == num_dims - 2) && (adjusted_shifts[i] % tile_dim) != 0) {
                    native_ok = false;
                    break;
                }
            }
        }

        if (native_ok) {
            for (size_t i = 0; i < adjusted_shifts.size(); ++i) {
                int dim = input_dims[i];
                if (dim < 0) {
                    dim += num_dims;
                }
                // adjusted_shifts[i] is already normalized to [0, shape[dim]).
                const int shift = adjusted_shifts[i];
                if (shift == 0) {
                    continue;
                }
                result = ttnn::prim::roll_sharded(
                    result, static_cast<uint32_t>(shift), static_cast<int32_t>(dim), native_mem_config);
            }
            // Apply requested output memory config if different from the input's.
            if (output_mem_config != native_mem_config) {
                result = ttnn::to_memory_config(result, output_mem_config, std::nullopt);
            }
            return result;
        }

        // Sub-tile rotation must move elements inside tiles: untilize, roll, tilize, all
        // staying sharded in L1.
        ttnn::Tensor rm = ttnn::untilize(input_tensor, native_mem_config);
        ttnn::Tensor rolled = roll(rm, shifts, input_dims);
        ttnn::Tensor retiled = ttnn::tilize(rolled, native_mem_config, input_tensor.dtype());
        if (output_mem_config != native_mem_config) {
            return ttnn::to_memory_config(retiled, output_mem_config, std::nullopt);
        }
        return retiled;
    }

    for (size_t i = 0; i < adjusted_shifts.size(); ++i) {
        int dim = input_dims[i];

        if (dim < 0) {
            dim += num_dims;
        }

        // adjusted_shifts[i] is already normalized to [0, shape[dim]).
        const int shift = adjusted_shifts[i];
        if (shift == 0) {
            continue;
        }

        ttsl::SmallVector<int> start_left(num_dims, 0), end_left;
        ttsl::SmallVector<int> start_right(num_dims, 0), end_right;

        for (int j = 0; j < num_dims; ++j) {
            end_left.push_back(size[j]);
            end_right.push_back(size[j]);
        }

        start_left[dim] = size[dim] - shift;
        start_right[dim] = 0;
        end_right[dim] = size[dim] - shift;

        ttnn::Tensor left_part = ttnn::slice(result, start_left, end_left, stride_vector);
        ttnn::Tensor right_part = ttnn::slice(result, start_right, end_right, stride_vector);

        std::vector<ttnn::Tensor> tensors_to_concat = {left_part, right_part};
        result = ttnn::concat(tensors_to_concat, dim);
    }

    if (output_mem_config != result.memory_config()) {
        result = ttnn::to_memory_config(result, output_mem_config, std::nullopt);
    }
    return result;
}

ttnn::Tensor roll(const ttnn::Tensor& input_tensor, const int shift, const std::optional<MemoryConfig>& memory_config) {
    // The flatten reshape to [1, total_elements] does not preserve sharding.
    TT_FATAL(
        !input_tensor.is_sharded(),
        "ttnn::roll without dims does not support sharded inputs. Convert to interleaved first.");

    ttsl::SmallVector<int> shifts = {shift};
    ttsl::SmallVector<int> dims = {1};  // Rolling will happen on dimension 1 after flattening

    auto original_shape = input_tensor.logical_shape();

    // Calculate total number of elements for flattening
    int total_elements = 1;
    for (int i = 0; i < original_shape.rank(); ++i) {
        total_elements *= original_shape[i];
    }

    // Flatten the input tensor to shape [1, total_elements]
    ttnn::Tensor result = ttnn::reshape(input_tensor, ttnn::Shape({1, total_elements}));

    result = roll(result, shifts, dims, memory_config);
    // Reshape back to the original shape
    result = ttnn::reshape(result, ttnn::Shape(original_shape));

    return result;
}

ttnn::Tensor roll(
    const ttnn::Tensor& input_tensor,
    const int shift,
    const int dim,
    const std::optional<MemoryConfig>& memory_config) {
    ttsl::SmallVector<int> shifts = {shift};
    ttsl::SmallVector<int> dims = {dim};

    return roll(input_tensor, shifts, dims, memory_config);
}

}  // namespace ttnn
