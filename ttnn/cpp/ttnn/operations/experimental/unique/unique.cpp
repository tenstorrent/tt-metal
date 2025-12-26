// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "unique.hpp"

#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "unique_common.hpp"

#include "device/unique_device_op.hpp"
#include "device/unique_device_op_types.hpp"

#include "tt-metalium/shape.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/clone/clone.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/sort/sort.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/tensor/types.hpp"

#include <numeric>

namespace ttnn::operations::experimental {

using namespace unique::common;
using unique::UniqueCB;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

struct UniquePreprocessingResult {
    Tensor preprocessed_input_tensor{};
    Tensor first_occurrences_tensor{};
    uint32_t single_fetch_elements_number{};
};

void validate_inputs(const Tensor& input, const std::optional<int32_t>& dim) {
    const auto input_rank = input.logical_shape().rank();
    if (dim.has_value()) {
        TT_ASSERT(
            *dim >= -input_rank, "dim must be at least -input_rank (dim: {}, -input_rank: {})", *dim, -input_rank);
    }
}

void calculate_and_insert_max_fetch_size(UniquePreprocessingResult& unique_preprocessing_result) {
    const auto& input_tensor = unique_preprocessing_result.preprocessed_input_tensor;
    const auto& first_occurrences_tensor = unique_preprocessing_result.first_occurrences_tensor;
    const auto l1_size_per_core = input_tensor.device()->l1_size_per_core();
    const auto confidence_margin = 0.8f;
    const auto input_datum_size = input_tensor.element_size();
    const auto input_compare_datum_size = input_datum_size;
    const auto first_occurrences_datum_size = first_occurrences_tensor.element_size();
    const auto result_datum_size = input_datum_size;
    const auto output_datum_size = input_datum_size;

    unique_preprocessing_result.single_fetch_elements_number = static_cast<uint32_t>(
        l1_size_per_core * confidence_margin /
        (input_datum_size + input_compare_datum_size + first_occurrences_datum_size * 3 + result_datum_size +
         output_datum_size));
}

UniquePreprocessingResult unique_preprocessing(const Tensor& input_tensor, const std::optional<int32_t>& dim) {
    UniquePreprocessingResult unique_preprocessing_result;
    unique_preprocessing_result.preprocessed_input_tensor = input_tensor;
    unique_preprocessing_result.first_occurrences_tensor =
        ttnn::zeros_like(input_tensor, FIRST_OCCURRENCES_TENSOR_DATA_TYPE);

    const uint32_t input_size = input_tensor.logical_shape().volume();

    calculate_and_insert_max_fetch_size(unique_preprocessing_result);

    if (unique_preprocessing_result.preprocessed_input_tensor.layout() != OUTPUT_TENSOR_LAYOUT) {
        unique_preprocessing_result.preprocessed_input_tensor =
            ttnn::to_layout(unique_preprocessing_result.preprocessed_input_tensor, OUTPUT_TENSOR_LAYOUT);
    }

    if (input_tensor.logical_shape().rank() > OUTPUT_TENSOR_RANK) {
        unique_preprocessing_result.preprocessed_input_tensor =
            ttnn::reshape(unique_preprocessing_result.preprocessed_input_tensor, Shape{input_size});
    }

    return unique_preprocessing_result;
}

Tensor unique_postprocessing(const std::vector<Tensor>& output_tensors) {
    const auto& output_tensor = output_tensors[0];
    const auto output_size_tensor_cpu = output_tensors[1].cpu();
    const auto output_size_tensor_host_buffer = tt::tt_metal::host_buffer::get_as<uint32_t>(output_size_tensor_cpu);
    const uint32_t size = output_size_tensor_host_buffer[0];
    std::cout << "SIZEEEEE " << size << std::endl;
    return ttnn::slice(
        output_tensor,
        ttnn::SmallVector<uint32_t>{0},
        ttnn::SmallVector<uint32_t>{size},
        ttnn::SmallVector<uint32_t>{1});
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

Tensor UniqueOperation::invoke(
    const Tensor& input,
    const bool& sorted,
    const bool& return_inverse,
    const bool& return_counts,
    const std::optional<int32_t>& dim,
    const std::optional<MemoryConfig>& memory_config) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    validate_inputs(input, dim);

    const auto unique_preprocessing_result = unique_preprocessing(input, dim);

    auto output_tensors = ttnn::prim::unique(
        unique_preprocessing_result.preprocessed_input_tensor,
        unique_preprocessing_result.first_occurrences_tensor,
        unique_preprocessing_result.single_fetch_elements_number,
        sorted,
        return_inverse,
        return_counts,
        dim,
        memory_config);

    return unique_postprocessing(output_tensors);
}

}  // namespace ttnn::operations::experimental
