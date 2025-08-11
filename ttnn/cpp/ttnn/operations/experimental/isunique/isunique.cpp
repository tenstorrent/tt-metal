// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "isunique.hpp"

#include "isunique_common.hpp"

#include "device/isunique_device_op.hpp"
#include "device/isunique_device_op_types.hpp"

#include "tt-metalium/shape.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/sort/sort.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/tensor/enum_types.hpp"

#include <magic_enum/magic_enum.hpp>
#include <numeric>

#include <bits/stdc++.h>

namespace ttnn::operations::experimental {

using namespace isunique::common;
using isunique::IsUniqueCB;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

void validate_inputs(const Tensor& input_tensor, const std::optional<int32_t>& dim) {
    const auto& input_shape = input_tensor.logical_shape();
    TT_FATAL();
}

void check_support(const Tensor& input_tensor, const std::optional<int32_t>& dim) {
    //
}

inline uint32_t sum_datum_sizes(
    const Tensor& input_tensor,
    const Tensor& index_hint_tensor,
    const std::optional<Tensor>& first_occurrences_tensor) {
    return first_occurrences_tensor.has_value() ? (input_tensor.element_size() + index_hint_tensor.element_size())
                                                : (input_tensor.element_size() + index_hint_tensor.element_size() +
                                                   first_occurrences_tensor->element_size());
}

// ONLY IN THE `dim==None` case
// calculate maximal possible allowance for a row of a SINGLE TENSOR being processed, it goes as follows:
// compute l1 memory with a safety margin
// sum all dtype sizes from all rows that x
// template <DataType... dtypes>
inline uint32_t calculate_max_row_size(
    const Tensor& input_tensor,
    const Tensor& index_hint_tensor,
    const std::optional<Tensor>& first_occurrences_tensor) {
    const uint32_t l1_size_per_core = input_tensor.device()->l1_size_per_core();
    constexpr double confidence_margin = 0.8;  // TODO(jbbieniekTT): subject to tuning
    const auto datums_sizes_sum = sum_datum_sizes(input_tensor, index_hint_tensor, first_occurrences_tensor);
    return static_cast<uint32_t>(l1_size_per_core * confidence_margin / datums_sizes_sum);
}

// template <DataType... dtypes>
std::tuple<double, double, double> compute_cost(
    const uint64_t& input_volume, const uint64_t& row_count, const uint32_t& cores, const bool& first_occurrences) {
    const double row_size = static_cast<double>(input_volume) / static_cast<double>(row_count);
    // Regular cost (direct)
    const double reg_direct =
        row_count * row_size * std::log2(row_size) / cores + row_count * row_size * (row_count - 1) / cores;
    // Regular cost (simplified)
    // const double reg_simplified = static_cast<double>(input_volume) * (std::log2(static_cast<double>(input_volume) /
    // row_count) + (row_count - 1)) / cores;

    // DRAM cost
    constexpr double dram_overhead_multiplier = 1000.;  // TODO(jbbieniekTT): subject to tuning
    const double dram = dram_overhead_multiplier * row_count * (row_count + 1) / 2 * (first_occurrences ? 3 : 2);

    return {reg_direct, dram, reg_direct + dram};
}

// the goal here is, given a continuous row-major tensor, as well as the following complexity functions of isunique with
// helper variables:

// row_size = calculate_max_row_size(input_tensor, first_occurrences)
// ^=>
// ^=>
// num_rows = input_volume ~/ row_size
// processed_pairs = num_rows * (num_rows - 1) / 2
// compute cost: O(row_size * log2(row_size)) for INITIAL sorting + O(num_rows * (num_rows - 1) / 2) pairs of rows
// processed * O(2 * row_size) for a pair of SORTED rows being compared
// ^=> compute cost = O(row_size * log2(row_size) + num_rows * (num_rows - 1) * row_size)
// ^=> compute cost = O(quicksort(row_size) + input_volume * num_rows)
// DRAM cost: O(num_rows * (num_rows - 1) / 2) for row swaps * O(get_simultaneous_rows(first_occurrences))
// template <DataType... dtypes>
OptimalHeuristic compute_optimal_heuristic(
    const Tensor& input_tensor,
    const Tensor& index_hint_tensor,
    const std::optional<Tensor>& first_occurrences_tensor) {
    const auto num_cores = input_tensor.device()->compute_with_storage_grid_size().x *
                           input_tensor.device()->compute_with_storage_grid_size().y;
    const auto input_volume = input_tensor.logical_volume();
    const auto MAX_ROW_SIZE = calculate_max_row_size(input_tensor, index_hint_tensor, first_occurrences_tensor);
    const uint64_t X_min = (input_volume + MAX_ROW_SIZE - 1) / MAX_ROW_SIZE;
    std::vector<uint64_t> candidates{};

    if (X_min <= num_cores) {
        for (int32_t b = 0; b <= std::log2(num_cores); ++b) {
            const uint32_t x = (1ULL << b);
            if (x >= std::max<uint32_t>(1, X_min)) {
                candidates.push_back(x);
            }
        }
    } else {
        // Next five multiples of 64
        const uint32_t start = (X_min + num_cores - 1) / num_cores;
        for (int32_t k = 0; k < (std::log2(num_cores) - 1); ++k) {
            candidates.push_back(num_cores * (start + k));
        }
    }

    OptimalHeuristic best{std::numeric_limits<double>::infinity(), input_volume, 0, 0, 0};

    for (const auto& X : candidates) {
        const uint32_t cores = X <= num_cores ? X : num_cores;
        const double Y = static_cast<double>(input_volume) / X;
        if (Y > MAX_ROW_SIZE) {
            continue;
        }

        const auto sum_of_datum_sizes = sum_datum_sizes(input_tensor, index_hint_tensor, first_occurrences_tensor);
        const auto [reg, dram, total] = compute_cost(sum_of_datum_sizes, input_volume, X, cores);
        if (total < best.total_cost) {
            best.total_cost = total;
            best.num_cores = cores;
        }
    }

    best.fill_value = input_volume / input_tensor.element_size() + 1;
    return best;
}

struct IsUniquePreprocessingResult {
    Tensor input_tensor;
    Tensor index_hint_tensor;
    std::optional<Tensor> first_occurrences_tensor;
    OptimalHeuristic optimal_heuristic;
};

IsUniquePreprocessingResult isunique_preprocessing(
    const QueueId& queue_id,
    const Tensor& input_tensor,
    const std::optional<int32_t>& dim,
    const bool& first_occurrences) {
    auto preprocessed_input_tensor = input_tensor;
    const auto input_volume = preprocessed_input_tensor.logical_volume();
    const auto input_datum_size = input_tensor.element_size();
    IsUniquePreprocessingResult is_unique_preprocwssing_result;

    if (dim.has_value()) {
        // sort along the isunique'd dim
        auto sort_output = ttnn::sort(queue_id, preprocessed_input_tensor, *dim, false, false);
        // sort_output[0] = ;
        is_unique_preprocwssing_result.input_tensor = std::move(sort_output.at(0));
        // the index tensor is a hint to the post-processing routine to reorder output values to the original order
        is_unique_preprocwssing_result.index_hint_tensor = std::move(sort_output.at(1));
    } else {
        // row-major O(input_volume) transform, which allows for a virtual reshape
        if (preprocessed_input_tensor.layout() != Layout::ROW_MAJOR) {
            preprocessed_input_tensor = ttnn::to_layout(preprocessed_input_tensor, Layout::ROW_MAJOR);
        }
        const ttnn::Shape optimal_shape = {is_unique_preprocwssing_result.optimal_heuristic.num};
        is_unique_preprocwssing_result.optimal_heuristic.fill_value = (input_volume / input_datum_size) + 1;
        is_unique_preprocwssing_result.index_hint_tensor = ttnn::full_like(
            preprocessed_input_tensor,
            is_unique_preprocwssing_result.optimal_heuristic.fill_value,
            PREDEFINED_TENSOR_DTYPES.at(IsUniqueCB::INDEX_HINT));
        is_unique_preprocwssing_result.first_occurrences_tensor =
            first_occurrences ? ttnn::full_like(
                                    preprocessed_input_tensor,
                                    is_unique_preprocwssing_result.optimal_heuristic.fill_value,
                                    PREDEFINED_TENSOR_DTYPES.at(IsUniqueCB::FIRST_OCCURRENCES))
                              : std::nullopt;
        is_unique_preprocwssing_result.optimal_heuristic = compute_optimal_heuristic(
            preprocessed_input_tensor,
            is_unique_preprocwssing_result.index_hint_tensor,
            is_unique_preprocwssing_result.first_occurrences_tensor);
    }

    return is_unique_preprocwssing_result;
}

Tensor isunique_postprocessing(
    const Tensor& output_tensor,
    const Layout& original_layout,
    const Shape& original_shape,
    const std::optional<int32_t>& dim,
    const IsUniquePreprocessingResult& conf) {
    if (original_layout != Layout::ROW_MAJOR) {
        //
    }
    //
    //
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

Tensor IsUniqueOperation::invoke(
    const QueueId& queue_id,
    const Tensor& input_tensor,
    const std::optional<int32_t>& dim,
    const bool& invert,
    const bool& first_occurrences,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& opt_out) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    validate_inputs(input_tensor, dim);
    check_support(input_tensor, dim);

    const auto optimal_isunique_configuration = isunique_preprocessing(queue_id, input_tensor, dim, first_occurrences);
    const Tensor output_tensor = ttnn::prim::isunique(
        optimal_isunique_configuration.input_tensor,
        optimal_isunique_configuration.index_hint_tensor,
        invert,
        dim,
        optimal_isunique_configuration.first_occurrences_tensor,
        optimal_isunique_configuration.fill_value,
        first_occurrences,
        memory_config);

    return isunique_postprocessing(output_tensor, dim, optimal_isunique_configuration);
}

}  // namespace ttnn::operations::experimental
