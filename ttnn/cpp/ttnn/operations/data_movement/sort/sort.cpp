// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sort.hpp"
#include "device/sort_device_operation.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/reduction/reduction_common/reduction_common.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"

#include <numeric>

namespace ttnn::operations::data_movement {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

uint32_t next_power_of_two(uint32_t n) {
    if (n <= 1) {
        return 1;
    }

    // If n is already a power of two, return it
    if ((n & (n - 1)) == 0) {
        return n;
    }

    // Otherwise, compute the next power of two
    uint32_t power = 1;
    while (power < n) {
        power <<= 1;
    }
    return power;
}

// pre_sort_transform_tensor
//
// Prepares the input tensor for the sort device op:
//   1. Transpose so the sort dimension becomes the last dimension.
//   2. Reshape to 4D.
//   3. For TILE layout: fill implicit tile-row padding with ±inf.
//      For ROW_MAJOR layout: pad the H dimension so that
//      combined_h (shape[0]*shape[1]*shape[2]) is a multiple of TILE_HEIGHT.
//   4. Pad the last dimension to the next power-of-two ≥ 2×TILE_WIDTH so the
//      bitonic-sort kernel receives a valid Wt.
Tensor pre_sort_transform_tensor(
    const Tensor& input_tensor,
    const int8_t dim,
    const bool is_dim_last_idx,
    const bool is_rank_le_4d,
    const bool descending) {
    if (input_tensor.logical_shape() == ttnn::Shape{1}) {
        return input_tensor;
    }

    const Tensor transposed_tensor = reduction_common::perform_transpose(input_tensor, is_dim_last_idx, dim, -1);
    const Tensor transformed_tensor = reduction_common::transform_to_4d_tensor(transposed_tensor, is_rank_le_4d);

    Tensor padded_tensor = transformed_tensor;
    const bool is_row_major = (transformed_tensor.layout() == Layout::ROW_MAJOR);

    if (!is_row_major) {
        // TILE layout: fill the implicit tile-row padding so the bitonic sort
        // ignores it (pads with ±inf).
        padded_tensor = ttnn::fill_implicit_tile_padding(
            transformed_tensor,
            descending ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity());
    } else {
        // ROW_MAJOR: the kernel processes rows in groups of TILE_HEIGHT (32).
        // If combined_h (= shape[0]*shape[1]*shape[2]) is not a multiple of
        // TILE_HEIGHT, pad the H dimension so every group is complete.
        const auto lshape_4d = padded_tensor.logical_shape();
        const uint32_t combined_h = lshape_4d[0] * lshape_4d[1] * lshape_4d[2];
        if (combined_h % tt::constants::TILE_HEIGHT != 0) {
            // Compute the per-H alignment: smallest k such that nc*k is a
            // multiple of TILE_HEIGHT, i.e. k = TILE_HEIGHT / gcd(TILE_HEIGHT, nc).
            const uint32_t nc = lshape_4d[0] * lshape_4d[1];
            const uint32_t gcd_val = std::gcd(tt::constants::TILE_HEIGHT, nc);
            const uint32_t h_alignment = tt::constants::TILE_HEIGHT / gcd_val;
            const uint32_t new_h = (lshape_4d[2] + h_alignment - 1) / h_alignment * h_alignment;
            padded_tensor = ttnn::pad(
                padded_tensor,
                ttnn::Array4D({lshape_4d[0], lshape_4d[1], new_h, lshape_4d[3]}),
                ttnn::Array4D({0, 0, 0, 0}),
                descending ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity(),
                /*use_multicore=*/true);
        }
    }

    // Pad the last dimension to the next power-of-two ≥ 2×TILE_WIDTH.
    // For TILE format use padded_shape (which includes implicit alignment);
    // for ROW_MAJOR use logical_shape (they are identical).
    const auto current_shape = is_row_major ? padded_tensor.logical_shape() : padded_tensor.padded_shape();
    const auto last_dim = current_shape[-1];
    auto padded_last_dim = next_power_of_two(last_dim);
    if ((padded_last_dim == last_dim) && (last_dim > tt::constants::TILE_WIDTH)) {
        return padded_tensor;
    }
    if (padded_last_dim == tt::constants::TILE_WIDTH) {
        padded_last_dim = tt::constants::TILE_WIDTH * 2;
    }

    const auto& padded_logical_shape = padded_tensor.logical_shape();
    return ttnn::pad(
        padded_tensor,
        ttnn::Array4D({padded_logical_shape[0], padded_logical_shape[1], padded_logical_shape[2], padded_last_dim}),
        ttnn::Array4D({0, 0, 0, 0}),
        descending ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity(),
        /*use_multicore=*/true);
}

std::vector<Tensor> post_sort_transform_tensor(
    const Tensor& input_tensor,
    std::vector<Tensor>& result,
    const int8_t dim,
    const bool is_dim_last_idx,
    const Shape& original_lshape,
    const MemoryConfig& input_memory_config) {
    // Reverse the pre-sort transformations
    const auto& input_shape = input_tensor.logical_shape();
    const auto orig_rank = input_shape.rank();

    // Check if manual W-padding was applied (for the power-of-two Wt requirement).
    const auto output_logical_shape = result[0].logical_shape();

    const int8_t normalized_dim = dim < 0 ? orig_rank + dim : dim;
    if (output_logical_shape[-1] != original_lshape[normalized_dim]) {
        const ttsl::SmallVector<uint32_t> step = {1, 1, 1, 1};
        const ttsl::SmallVector<uint32_t> start_index = {0, 0, 0, 0};
        const ttsl::SmallVector<uint32_t> end_index = {
            output_logical_shape[-4],
            output_logical_shape[-3],
            output_logical_shape[-2],
            original_lshape[normalized_dim]};
        result[0] = ttnn::slice(result[0], start_index, end_index, step, input_memory_config);
        result[1] = ttnn::slice(result[1], start_index, end_index, step, input_memory_config);
    }

    // Slice H back if it was padded for ROW_MAJOR (combined_h was made a multiple of TILE_HEIGHT).
    // original_combined_h = total elements / sort-dim size, i.e. the product of all non-sort dims.
    {
        const auto cur_shape = result[0].logical_shape();
        const uint32_t result_combined_h = cur_shape[0] * cur_shape[1] * cur_shape[2];
        const uint32_t original_combined_h =
            static_cast<uint32_t>(original_lshape.volume()) / original_lshape[normalized_dim];
        if (result_combined_h != original_combined_h) {
            const uint32_t nc = cur_shape[0] * cur_shape[1];
            const uint32_t h_sliced = original_combined_h / nc;
            const ttsl::SmallVector<uint32_t> step = {1, 1, 1, 1};
            const ttsl::SmallVector<uint32_t> start_index = {0, 0, 0, 0};
            const ttsl::SmallVector<uint32_t> end_index = {cur_shape[0], cur_shape[1], h_sliced, cur_shape[3]};
            result[0] = ttnn::slice(result[0], start_index, end_index, step, input_memory_config);
            result[1] = ttnn::slice(result[1], start_index, end_index, step, input_memory_config);
        }
    }

    // Reshape back to original rank if needed
    if (orig_rank < 4 && orig_rank > 1) {
        result[0] = ttnn::squeeze_from_4D(result[0], orig_rank);
        result[1] = ttnn::squeeze_from_4D(result[1], orig_rank);
    } else if (orig_rank == 1) {
        const ttsl::SmallVector<uint32_t> result_shape(input_shape.cbegin(), input_shape.cend());
        result[0] = ttnn::reshape(result[0], ttnn::Shape{result_shape});
        result[1] = ttnn::reshape(result[1], ttnn::Shape{result_shape});
    } else if (orig_rank > 4) {
        // The 4D `result` tensor came from squeeze_from_ND_to_4D applied to the
        // *transposed* high-rank tensor (the sort dim was moved to the last
        // position by `perform_transpose`).  To invert correctly we must reshape
        // back to the transposed N-D shape — NOT the original — so that the
        // subsequent `transpose` swaps it back to the user's layout.
        //
        // Targeting the transposed shape keeps the last dim unchanged, which
        // triggers ttnn::reshape's `this_is_view` fast path (pure metadata via
        // ttnn::experimental::view) and entirely bypasses the device reshape
        // kernel.  This is essential for UINT16 indices, which the device
        // reshape kernel rejects, and avoids the cost of a dtype round-trip.
        ttsl::SmallVector<uint32_t> reshape_target(input_shape.cbegin(), input_shape.cend());
        if (!is_dim_last_idx) {
            const int normalized = dim < 0 ? orig_rank + dim : dim;
            std::swap(reshape_target[normalized], reshape_target[orig_rank - 1]);
        }
        result[0] = ttnn::reshape(result[0], ttnn::Shape{reshape_target});
        result[1] = ttnn::reshape(result[1], ttnn::Shape{reshape_target});
    }

    // Transpose back to original dimension order if needed
    if (!is_dim_last_idx) {
        result[0] = ttnn::transpose(result[0], dim, -1, input_tensor.memory_config());
        result[1] = ttnn::transpose(result[1], dim, -1, input_tensor.memory_config());
    }

    TT_FATAL(
        result[0].logical_shape() == original_lshape,
        "Output tensor transformation did not create correct output shape! Got: {}, expected: {}",
        result[0].logical_shape(),
        original_lshape);

    return result;
}

bool validate_optional_output_tensors_for_early_exit(
    const std::optional<std::tuple<Tensor, Tensor>>& optional_output_tensors, const Shape& original_lshape) {
    if (!optional_output_tensors.has_value()) {
        return false;
    }

    auto output_tensor_0 = std::get<0>(optional_output_tensors.value());
    auto output_tensor_1 = std::get<1>(optional_output_tensors.value());

    return output_tensor_0.logical_shape() == original_lshape && output_tensor_1.logical_shape() == original_lshape;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

}  // namespace ttnn::operations::data_movement

namespace ttnn {

std::vector<Tensor> sort(
    const Tensor& input_tensor,
    const int8_t dim,
    const bool descending,
    const bool stable,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<std::tuple<Tensor&, Tensor&>> optional_output_tensors) {
    TT_FATAL(!stable, "ttnn::sort: stable=True is not yet implemented.");

    const ttnn::Shape& original_lshape = input_tensor.logical_shape();
    const auto rank = input_tensor.logical_shape().rank();

    // FLOAT32 inputs require UINT32 indices (device-side validation enforces this for the
    // non-early-exit path; keep early exits consistent).
    const DataType index_dtype = (input_tensor.dtype() == DataType::FLOAT32) ? DataType::UINT32 : DataType::UINT16;

    // Check for early exit for scalar or empty tensors tensors
    if ((original_lshape == ttnn::Shape{}) || (original_lshape == ttnn::Shape{1})) {
        auto indices = ttnn::zeros_like(input_tensor, index_dtype);
        if (operations::data_movement::CMAKE_UNIQUE_NAMESPACE::validate_optional_output_tensors_for_early_exit(
                optional_output_tensors, original_lshape)) {
            std::get<0>(*optional_output_tensors) = input_tensor;
            std::get<1>(*optional_output_tensors) = indices;
            return {std::get<0>(optional_output_tensors.value()), std::get<1>(optional_output_tensors.value())};
        }
        return {input_tensor, indices};
    }

    TT_FATAL(
        dim >= -static_cast<int8_t>(rank) && dim < static_cast<int8_t>(rank),
        "Sort dim {} is out of range for rank-{} tensor",
        dim,
        rank);

    const int32_t normalized_dim = dim < 0 ? static_cast<int32_t>(rank) + dim : dim;
    if (original_lshape[normalized_dim] == 1) {
        auto indices = ttnn::zeros_like(input_tensor, index_dtype);
        if (operations::data_movement::CMAKE_UNIQUE_NAMESPACE::validate_optional_output_tensors_for_early_exit(
                optional_output_tensors, original_lshape)) {
            std::get<0>(*optional_output_tensors) = input_tensor;
            std::get<1>(*optional_output_tensors) = indices;
            return {std::get<0>(optional_output_tensors.value()), std::get<1>(optional_output_tensors.value())};
        }
        return {input_tensor, indices};
    }

    const bool is_dim_last_idx = (dim == -1 || dim == rank - 1);
    const bool is_rank_le_4d = rank <= 4;

    namespace dm = operations::data_movement::CMAKE_UNIQUE_NAMESPACE;

    // Determine the user-requested output memory config.
    // Priority: preallocated out= tensors > explicit memory_config arg > input tensor.
    // When preallocated outputs have different memory configs, sort to the values config
    // and convert the index output separately afterwards.
    MemoryConfig sort_mem_cfg = memory_config.value_or(input_tensor.memory_config());
    std::optional<MemoryConfig> index_mem_cfg_override;
    if (optional_output_tensors.has_value()) {
        const auto& out0_mem = std::get<0>(*optional_output_tensors).memory_config();
        const auto& out1_mem = std::get<1>(*optional_output_tensors).memory_config();
        sort_mem_cfg = out0_mem;
        if (out0_mem != out1_mem) {
            index_mem_cfg_override = out1_mem;
        }
    }

    // Convert sharded inputs to DRAM before the pre-sort transforms.
    // ttnn::pad (called inside pre_sort_transform_tensor) does not support
    // sharded tensors and will hang or produce incorrect results if the input
    // is still sharded when it is called.  We capture the user-requested output
    // memory config in sort_mem_cfg above before doing this conversion so that
    // the output can be converted back to the sharded config after sorting.
    const Tensor transform_input = input_tensor.memory_config().is_sharded()
                                       ? ttnn::to_memory_config(input_tensor, ttnn::DRAM_MEMORY_CONFIG)
                                       : input_tensor;

    Tensor padded_input_tensor =
        dm::pre_sort_transform_tensor(transform_input, dim, is_dim_last_idx, is_rank_le_4d, descending);
    const MemoryConfig device_op_mem_cfg = sort_mem_cfg.is_sharded() ? ttnn::DRAM_MEMORY_CONFIG : sort_mem_cfg;

    // Canonicalize any preallocated output tensors.
    // Only pass them to the device op when their layout matches the padded input.
    // If there is a mismatch (e.g. TILE input with RM preallocated outputs), the
    // TILE program factory would write tile-formatted data into an RM buffer and
    // produce corrupt results.  In that case we sort to the device-default output
    // and convert the layout afterwards before rebinding the user's handles.
    std::vector<std::optional<Tensor>> output_tensors{std::nullopt, std::nullopt};
    bool preallocated_layout_mismatch = false;
    if (optional_output_tensors.has_value()) {
        const auto& out0 = std::get<0>(*optional_output_tensors);
        const auto& out1 = std::get<1>(*optional_output_tensors);
        const bool values_layout_match = (out0.layout() == padded_input_tensor.layout());
        const bool indices_layout_match = (out1.layout() == padded_input_tensor.layout());
        preallocated_layout_mismatch = !values_layout_match || !indices_layout_match;
        if (!preallocated_layout_mismatch) {
            const auto canonicalize_output = [&](const Tensor& t) {
                return dm::pre_sort_transform_tensor(t, dim, is_dim_last_idx, is_rank_le_4d, descending);
            };
            output_tensors[0] = canonicalize_output(out0);
            output_tensors[1] = canonicalize_output(out1);
        }
    }

    // pre_sort_transform_tensor always moves the sort dimension to position -1,
    // so the device op always sorts along the last dimension.
    auto sorted_tensors = ttnn::prim::sort(
        padded_input_tensor, static_cast<int8_t>(-1), descending, stable, device_op_mem_cfg, output_tensors);

    auto results = dm::post_sort_transform_tensor(
        input_tensor, sorted_tensors, dim, is_dim_last_idx, original_lshape, device_op_mem_cfg);

    // The device op always writes to DRAM when sort_mem_cfg is sharded (to avoid
    // shard-spec conflicts with the W-padded intermediate tensor shape).  After
    // the post-transform slice has restored the original W dimension, convert back
    // to the sharded config so the output memory layout matches the input.
    if (sort_mem_cfg.is_sharded() && !preallocated_layout_mismatch) {
        results[0] = ttnn::to_memory_config(results[0], sort_mem_cfg);
        results[1] = ttnn::to_memory_config(results[1], sort_mem_cfg);
    }

    // If pre-allocated outputs had a different layout from the device op output
    // (e.g. TILE input + RM preallocated outputs), convert the layout now so the
    // user's handle receives data in the expected format.  Also restore any sharded
    // memory config that was requested on the preallocated outputs.
    if (preallocated_layout_mismatch && optional_output_tensors.has_value()) {
        const Layout values_target = std::get<0>(*optional_output_tensors).layout();
        const Layout indices_target = std::get<1>(*optional_output_tensors).layout();
        if (results[0].layout() != values_target) {
            results[0] = ttnn::to_layout(results[0], values_target);
        }
        if (results[1].layout() != indices_target) {
            results[1] = ttnn::to_layout(results[1], indices_target);
        }
        if (sort_mem_cfg.is_sharded()) {
            results[0] = ttnn::to_memory_config(results[0], sort_mem_cfg);
            results[1] = ttnn::to_memory_config(results[1], sort_mem_cfg);
        }
    }

    // Apply independent memory config for the index output if the preallocated
    // values and indices tensors requested different memory configs.
    if (index_mem_cfg_override.has_value()) {
        results[1] = ttnn::to_memory_config(results[1], *index_mem_cfg_override);
    }

    // For preallocated outputs, rebind the user's handle when the underlying
    // buffer changed.
    if (optional_output_tensors.has_value()) {
        const auto rebind_if_changed = [](Tensor& user_t, const Tensor& result_t) {
            if (user_t.buffer() != result_t.buffer()) {
                user_t = result_t;
            }
        };
        rebind_if_changed(std::get<0>(*optional_output_tensors), results[0]);
        rebind_if_changed(std::get<1>(*optional_output_tensors), results[1]);
    }

    return results;
}

}  // namespace ttnn
