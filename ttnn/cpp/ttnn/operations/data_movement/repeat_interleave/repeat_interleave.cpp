// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "repeat_interleave.hpp"

#include <tt-metalium/constants.hpp>

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

// repeat interleave supports repeats as 1 to inf, dim between 0 to 2
//
// Named (not anonymous-call-site) so the last-dim branch below can recurse into it directly:
// recursing through the public ttnn::repeat_interleave() would re-enter the "auto" selector and
// let a forced implementation="native" call silently escalate to codegen mid-recursion.
Tensor repeat_interleave_native(
    const Tensor& input_a, uint32_t repeat, int32_t dim, const std::optional<MemoryConfig>& output_mem_config) {
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

// Rank-general TILE page geometry for the (outer, non-sub-tile) repeated axis --
// ops/repeat/spec.py::_page_map, shared byte-for-byte by repeat and repeat_interleave.
struct PageMap {
    uint32_t lower_pages;
    uint32_t rep_dim_pages;
    uint32_t total_out_pages;
    uint32_t stick_size;
};

PageMap tile_page_map(const Tensor& input, uint32_t rep_dim, uint32_t num_repeats) {
    const auto& shape = input.logical_shape();
    const uint32_t ndim = shape.rank();
    const uint32_t ht = (shape[ndim - 2] + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT;
    const uint32_t wt = (shape[ndim - 1] + tt::constants::TILE_WIDTH - 1) / tt::constants::TILE_WIDTH;
    std::vector<uint32_t> dim_pages;
    dim_pages.reserve(ndim);
    for (uint32_t i = 0; i + 2 < ndim; ++i) {
        dim_pages.push_back(shape[i]);
    }
    dim_pages.push_back(ht);
    dim_pages.push_back(wt);

    uint32_t lower_pages = 1;
    for (uint32_t d = rep_dim + 1; d < dim_pages.size(); ++d) {
        lower_pages *= dim_pages[d];
    }
    uint32_t volume_tiles = 1;
    for (uint32_t pages : dim_pages) {
        volume_tiles *= pages;
    }
    return {lower_pages, dim_pages[rep_dim], volume_tiles * num_repeats, /*stick_size=*/0};
}

// Rank-general RM (stick) page geometry for a whole-stick (outer/H) repeated axis --
// ops/repeat_interleave/spec.py::build_repeat_interleave_rm_factory's host math.
PageMap rm_page_map(const Tensor& input, uint32_t rep_dim, uint32_t num_repeats) {
    const auto& shape = input.logical_shape();
    const uint32_t ndim = shape.rank();
    std::vector<uint32_t> dim_pages;
    dim_pages.reserve(ndim);
    for (uint32_t i = 0; i + 1 < ndim; ++i) {
        dim_pages.push_back(shape[i]);
    }
    dim_pages.push_back(1);

    uint32_t lower_pages = 1;
    for (uint32_t d = rep_dim + 1; d < dim_pages.size(); ++d) {
        lower_pages *= dim_pages[d];
    }
    uint32_t total_src_pages = 1;
    for (uint32_t pages : dim_pages) {
        total_src_pages *= pages;
    }
    return {lower_pages, dim_pages[rep_dim], total_src_pages * num_repeats, shape[ndim - 1] * input.element_size()};
}

// operation_attributes_t.rep_dim's storage convention; recovered by
// repeat_interleave_codegen_device_operation.cpp and the codegen program factory.
constexpr uint32_t kRepDimPadRank = 4;

Tensor repeat_interleave_codegen_dispatch(
    const Tensor& input, uint32_t repeats, uint32_t normalized_dim, const MemoryConfig& mem_config) {
    const uint32_t ndim = input.logical_shape().rank();
    const uint32_t padded_dim = normalized_dim + (kRepDimPadRank - ndim);
    const PageMap page_map = input.layout() == Layout::TILE ? tile_page_map(input, normalized_dim, repeats)
                                                            : rm_page_map(input, normalized_dim, repeats);
    return ttnn::prim::repeat_interleave_codegen(
        input,
        padded_dim,
        repeats,
        page_map.lower_pages,
        page_map.rep_dim_pages,
        page_map.total_out_pages,
        page_map.stick_size,
        /*stick_size_out=*/0,
        mem_config);
}

}  // namespace

Tensor repeat_interleave(
    const Tensor& input_a,
    uint32_t repeats,
    int32_t dim,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::string& implementation) {
    using ttnn::operations::data_movement::ImplementationSelector;
    using ttnn::operations::data_movement::is_demoted;
    using ttnn::operations::data_movement::parse_implementation;
    using ttnn::operations::data_movement::supported_by_codegen;

    const ImplementationSelector selector = parse_implementation(implementation);

    if (selector == ImplementationSelector::Native) {
        return repeat_interleave_native(input_a, repeats, dim, output_mem_config);
    }

    if (selector == ImplementationSelector::Codegen) {
        TT_FATAL(
            supported_by_codegen(input_a, repeats, dim, output_mem_config),
            "repeat_interleave: implementation=\"codegen\" requested for an input/attribute "
            "combination supported_by_codegen() rejects");
        const uint32_t normalized_dim = input_a.logical_shape().get_normalized_index(dim);
        return repeat_interleave_codegen_dispatch(
            input_a, repeats, normalized_dim, output_mem_config.value_or(input_a.memory_config()));
    }

    if (supported_by_codegen(input_a, repeats, dim, output_mem_config) &&
        !is_demoted(input_a, repeats, dim, output_mem_config)) {
        const uint32_t normalized_dim = input_a.logical_shape().get_normalized_index(dim);
        return repeat_interleave_codegen_dispatch(
            input_a, repeats, normalized_dim, output_mem_config.value_or(input_a.memory_config()));
    }
    return repeat_interleave_native(input_a, repeats, dim, output_mem_config);
}

}  // namespace ttnn
