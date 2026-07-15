// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "repeat_interleave.hpp"

#include <array>

#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/codegen/repeat_interleave_codegen_device_operation.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/codegen/repeat_interleave_codegen_supported.hpp"
#include "ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {

namespace {

using ttnn::operations::data_movement::repeat_interleave::ImplementationSelector;
using ttnn::operations::data_movement::repeat_interleave::is_demoted;
using ttnn::operations::data_movement::repeat_interleave::parse_implementation;
using ttnn::operations::data_movement::repeat_interleave::supported_by_codegen;

// repeat interleave supports repeats as 1 to inf, dim between 0 to 2
ttnn::Tensor repeat_interleave_native(
    const ttnn::Tensor& input_a, uint32_t repeat, int32_t dim, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> combined_tensors;
    combined_tensors.reserve(repeat);
    MemoryConfig mem_config = output_mem_config.value_or(input_a.memory_config());
    if (repeat == 1) {
        return ttnn::to_memory_config(input_a, mem_config);
    }
    const auto& input_a_shape = input_a.logical_shape();
    uint32_t input_rank = input_a_shape.rank();
    uint32_t normalized_dim = input_a_shape.get_normalized_index(dim);
    if (normalized_dim == input_rank - 1) {
        auto transposed_input = ttnn::transpose(input_a, -1, -2, mem_config);
        // Recurse into the native implementation directly (not the public dispatcher) so the
        // "native" selector stays unconditionally native, never escalating to auto/codegen.
        auto repeated_input = repeat_interleave_native(transposed_input, repeat, -2, mem_config);
        return ttnn::transpose(repeated_input, -1, -2, mem_config);
    }

    ttnn::Tensor rm_input = input_a;
    bool typecast = input_a.dtype() != DataType::BFLOAT16;
    if (typecast) {
        rm_input = ttnn::typecast(rm_input, DataType::BFLOAT16, mem_config);
    }

    rm_input = ttnn::to_layout(rm_input, Layout::ROW_MAJOR);
    const auto& rm_input_shape = rm_input.logical_shape();
    ttsl::SmallVector<uint32_t> final_shape;
    final_shape.reserve(input_rank);
    for (uint32_t i = 0; i < rm_input_shape.rank(); i++) {
        final_shape.push_back(rm_input_shape[i]);
    }

    final_shape[normalized_dim] *= repeat;

    auto unsqueezed_tensor = ttnn::unsqueeze(rm_input, normalized_dim + 1);
    std::vector<Tensor> combined_tensors_batch;
    constexpr uint32_t repeats_batched = 32;
    combined_tensors_batch.reserve(std::min(repeat, repeats_batched));
    for (uint32_t i = 0; i < repeat; i++) {
        combined_tensors_batch.push_back(unsqueezed_tensor);

        // Concatenate every 32 tensors or at the end of the loop
        if (combined_tensors_batch.size() == repeats_batched || i == repeat - 1) {
            auto batch_concat = ttnn::concat(combined_tensors_batch, normalized_dim + 1);
            combined_tensors.push_back(batch_concat);
            combined_tensors_batch.clear();
        }
    }

    auto concatenated_tensor = ttnn::concat(combined_tensors, normalized_dim + 1);
    auto reshaped_tensor = ttnn::reshape(concatenated_tensor, ttnn::Shape(final_shape));
    auto original_layout = ttnn::to_layout(reshaped_tensor, input_a.layout());
    return typecast ? ttnn::typecast(original_layout, input_a.dtype(), mem_config) : original_layout;
}

// Host-side transliteration of the page-map math in ops/repeat_interleave/spec.py (and, for the
// TILE path, ops/repeat/spec.py's shared build_repeat_tile) that populates
// RepeatInterleaveCodegenParams. `normalized_dim` is already 0..ndim-1 (non-negative).
ttnn::prim::RepeatInterleaveCodegenParams build_codegen_params(
    const Tensor& input_tensor, uint32_t repeats, uint32_t normalized_dim, const MemoryConfig& mem_config) {
    const auto logical_shape = input_tensor.logical_shape();
    const uint32_t ndim = logical_shape.rank();
    const uint32_t pad = 4 - ndim;

    std::array<uint32_t, 4> shape4 = {1, 1, 1, 1};
    for (uint32_t i = 0; i < ndim; i++) {
        shape4[pad + i] = logical_shape[i];
    }
    const uint32_t rep_dim_4d = normalized_dim + pad;

    ttnn::prim::RepeatInterleaveCodegenParams params{};
    params.num_repeats = repeats;
    params.output_mem_config = mem_config;

    const bool row_major = input_tensor.layout() == Layout::ROW_MAJOR;
    const uint32_t elem_size = input_tensor.element_size();

    if (row_major && normalized_dim == ndim - 1) {
        // RM last-dim (within-stick W) path: rep_dim == 3 is the marker the program factory and
        // validate use to select this branch (the 4D-padded last dim is always index 3).
        const uint32_t w_in = shape4[3];
        params.rep_dim = 3;
        params.stick_size = w_in * elem_size;
        params.stick_size_out = w_in * repeats * elem_size;
        params.total_out_pages = shape4[0] * shape4[1] * shape4[2];
        return params;
    }

    std::array<uint32_t, 4> dim_pages = {1, 1, 1, 1};
    if (row_major) {
        // RM higher-dim (whole-stick, outer/H) path: pages are sticks.
        dim_pages = {shape4[0], shape4[1], shape4[2], 1};
        params.stick_size = shape4[3] * elem_size;
    } else {
        // TILE path: pages are tiles (ops/repeat/spec.py's _page_map).
        constexpr uint32_t kTileH = 32;
        constexpr uint32_t kTileW = 32;
        const uint32_t ht = (shape4[2] + kTileH - 1) / kTileH;
        const uint32_t wt = (shape4[3] + kTileW - 1) / kTileW;
        dim_pages = {shape4[0], shape4[1], ht, wt};
    }

    uint32_t lower_pages = 1;
    for (uint32_t d = rep_dim_4d + 1; d < 4; d++) {
        lower_pages *= dim_pages[d];
    }
    uint32_t total_pages = 1;
    for (uint32_t d = 0; d < 4; d++) {
        total_pages *= dim_pages[d];
    }

    params.rep_dim = rep_dim_4d;
    params.lower_pages = lower_pages;
    params.rep_dim_pages = dim_pages[rep_dim_4d];
    params.total_out_pages = total_pages * repeats;
    return params;
}

}  // namespace

ttnn::Tensor repeat_interleave(
    const ttnn::Tensor& input_a,
    uint32_t repeats,
    int32_t dim,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::string& implementation) {
    const MemoryConfig mem_config = output_mem_config.value_or(input_a.memory_config());
    const ImplementationSelector sel = parse_implementation(implementation);

    if (sel == ImplementationSelector::kNative) {
        return repeat_interleave_native(input_a, repeats, dim, output_mem_config);
    }

    const uint32_t normalized_dim = input_a.logical_shape().get_normalized_index(dim);

    if (sel == ImplementationSelector::kCodegen) {
        TT_FATAL(
            supported_by_codegen(input_a, repeats, static_cast<int32_t>(normalized_dim)),
            "ttnn.repeat_interleave: implementation=\"codegen\" is not supported for this input (dim={}, "
            "repeats={})",
            dim,
            repeats);
        const auto params = build_codegen_params(input_a, repeats, normalized_dim, mem_config);
        return ttnn::prim::repeat_interleave_codegen(input_a, params);
    }

    // Auto: codegen iff supported and not perf-demoted; else native.
    if (supported_by_codegen(input_a, repeats, static_cast<int32_t>(normalized_dim)) &&
        !is_demoted(input_a, repeats, static_cast<int32_t>(normalized_dim))) {
        const auto params = build_codegen_params(input_a, repeats, normalized_dim, mem_config);
        return ttnn::prim::repeat_interleave_codegen(input_a, params);
    }
    return repeat_interleave_native(input_a, repeats, dim, output_mem_config);
}

}  // namespace ttnn
