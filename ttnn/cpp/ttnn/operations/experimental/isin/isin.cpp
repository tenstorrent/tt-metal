// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "isin.hpp"

#include "isin_common.hpp"

#include "device/isin_device_op.hpp"
#include "device/isin_device_op_types.hpp"

#include "tt-metalium/shape.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/clone/clone.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/sort/sort.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/types.hpp"

#include <numeric>

namespace ttnn::operations::experimental {

using namespace isin::common;
using isin::IsInCB;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

constexpr DataType OUTPUT_DATA_TYPE = DataType::UINT8;

struct IsInPreprocessingResult {
    Tensor preprocessed_elements_tensor;
    Tensor preprocessed_test_elements_tensor;
    Shape original_elements_shape;
    Layout original_elements_layout;
    int mask_value;
    uint32_t elements_size;
    uint32_t test_elements_size;
    uint32_t single_fetch_elements_number;
};

void validate_inputs(const Tensor& elements, const Tensor& test_elements) {
    const auto& elements_shape = elements.logical_shape();
    const auto& test_elements_shape = test_elements.logical_shape();
}

uint32_t calculate_max_fetch_size(const Tensor& elements, const Tensor& test_elements) {
    const auto l1_size_per_core = elements.device()->l1_size_per_core();
    const auto confidence_margin = 0.8f;
    const auto elements_datum_size = elements.element_size();
    const auto test_elements_datum_size = test_elements.element_size();
    const auto output_datum_size = 4;

    return l1_size_per_core * confidence_margin / (elements_datum_size + test_elements_datum_size + output_datum_size);
    // return 1024;
}

IsInPreprocessingResult isin_preprocessing(
    const QueueId& queue_id, const Tensor& elements, const Tensor& test_elements) {
    IsInPreprocessingResult is_in_preprocessing_result;
    is_in_preprocessing_result.preprocessed_elements_tensor =
        ttnn::clone(elements, elements.dtype(), elements.memory_config(), std::nullopt);
    is_in_preprocessing_result.preprocessed_test_elements_tensor = test_elements;
    is_in_preprocessing_result.original_elements_shape = elements.logical_shape();
    is_in_preprocessing_result.original_elements_layout = elements.layout();
    is_in_preprocessing_result.elements_size = elements.logical_volume();
    is_in_preprocessing_result.test_elements_size = test_elements.logical_volume();
    is_in_preprocessing_result.single_fetch_elements_number = calculate_max_fetch_size(elements, test_elements);
    is_in_preprocessing_result.mask_value = static_cast<uint32_t>(-1);

    if (is_in_preprocessing_result.preprocessed_elements_tensor.layout() != OUTPUT_TENSOR_LAYOUT) {
        is_in_preprocessing_result.preprocessed_elements_tensor =
            ttnn::to_layout(is_in_preprocessing_result.preprocessed_elements_tensor, OUTPUT_TENSOR_LAYOUT);
    }
    if (is_in_preprocessing_result.preprocessed_test_elements_tensor.layout() != Layout::ROW_MAJOR) {
        is_in_preprocessing_result.preprocessed_test_elements_tensor =
            ttnn::to_layout(is_in_preprocessing_result.preprocessed_test_elements_tensor, Layout::ROW_MAJOR);
    }

    if (elements.logical_shape().rank() > OUTPUT_TENSOR_RANK) {
        is_in_preprocessing_result.preprocessed_elements_tensor = ttnn::reshape(
            is_in_preprocessing_result.preprocessed_elements_tensor, Shape{is_in_preprocessing_result.elements_size});
    }
    if (test_elements.logical_shape().rank() > 1) {
        is_in_preprocessing_result.preprocessed_test_elements_tensor = ttnn::reshape(
            is_in_preprocessing_result.preprocessed_test_elements_tensor,
            Shape{is_in_preprocessing_result.test_elements_size});
    }

    return is_in_preprocessing_result;
}

Tensor isin_postprocessing(Tensor& output_tensor, const IsInPreprocessingResult& is_in_preprocessing_result) {
    // output_tensor = ttnn::to_layout(output_tensor, Layout::TILE);
    // output_tensor = ttnn::typecast(output_tensor, DataType::UINT32);
    if (is_in_preprocessing_result.original_elements_shape.rank() != OUTPUT_TENSOR_RANK) {
        output_tensor = ttnn::reshape(output_tensor, is_in_preprocessing_result.original_elements_shape);
    }
    if (is_in_preprocessing_result.original_elements_layout != OUTPUT_TENSOR_LAYOUT) {
        output_tensor = ttnn::to_layout(output_tensor, is_in_preprocessing_result.original_elements_layout);
    }

    return output_tensor;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

Tensor IsInOperation::invoke(
    const QueueId& queue_id,
    const Tensor& elements,
    const Tensor& test_elements,
    const bool& assume_unique,
    const bool& invert,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& opt_out) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    validate_inputs(elements, test_elements);

    const auto is_in_preprocessing_result = isin_preprocessing(queue_id, elements, test_elements);

    Tensor output_tensor = ttnn::prim::isin(
        is_in_preprocessing_result.preprocessed_elements_tensor,
        is_in_preprocessing_result.preprocessed_test_elements_tensor,
        is_in_preprocessing_result.single_fetch_elements_number,
        assume_unique,
        invert,
        memory_config,
        opt_out,
        queue_id);

    return isin_postprocessing(output_tensor, is_in_preprocessing_result);
}

}  // namespace ttnn::operations::experimental
