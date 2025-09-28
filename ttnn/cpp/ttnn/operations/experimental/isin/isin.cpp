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
#include "ttnn/tensor/types.hpp"

#include <enchantum/enchantum.hpp>
#include <numeric>

namespace ttnn::operations::experimental {

using namespace isin::common;
using isin::IsInCB;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

struct IsInPreprocessingResult {
    Tensor preprocessed_elements_tensor{};
    Tensor preprocessed_test_elements_tensor{};
    Shape original_elements_shape{};
    Layout original_elements_layout{};
    uint32_t elements_size{};
    uint32_t test_elements_size{};
    uint32_t single_fetch_elements_number{};
};

void validate_inputs(const Tensor& elements, const Tensor& test_elements) {
    TT_FATAL(
        elements.dtype() == test_elements.dtype(),
        "elements.dtype: {}, test_elements.dtype: {}, they must be equal",
        enchantum::to_string(elements.dtype()),
        enchantum::to_string(test_elements.dtype()));
}

uint32_t calculate_max_fetch_size(const Tensor& elements, const Tensor& test_elements) {
    const auto l1_size_per_core = elements.device()->l1_size_per_core();
    const auto confidence_margin = 0.75f;
    const auto elements_datum_size = elements.element_size();
    const auto test_elements_datum_size = test_elements.element_size();
    const auto output_datum_size = elements_datum_size;

    return (static_cast<uint32_t>(
               l1_size_per_core * confidence_margin /
               (elements_datum_size + test_elements_datum_size + output_datum_size))) &
           static_cast<uint32_t>(~(63U));
}

IsInPreprocessingResult isin_preprocessing(const Tensor& elements, const Tensor& test_elements) {
    IsInPreprocessingResult is_in_preprocessing_result;
    is_in_preprocessing_result.preprocessed_elements_tensor = elements;
    is_in_preprocessing_result.preprocessed_test_elements_tensor = test_elements;
    is_in_preprocessing_result.original_elements_shape =
        is_in_preprocessing_result.preprocessed_elements_tensor.logical_shape();
    is_in_preprocessing_result.original_elements_layout =
        is_in_preprocessing_result.preprocessed_elements_tensor.layout();
    is_in_preprocessing_result.elements_size = is_in_preprocessing_result.preprocessed_elements_tensor.logical_volume();
    is_in_preprocessing_result.test_elements_size =
        is_in_preprocessing_result.preprocessed_test_elements_tensor.logical_volume();
    is_in_preprocessing_result.single_fetch_elements_number = calculate_max_fetch_size(
        is_in_preprocessing_result.preprocessed_elements_tensor,
        is_in_preprocessing_result.preprocessed_test_elements_tensor);

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
    const Tensor& elements,
    const Tensor& test_elements,
    bool assume_unique,
    bool invert,
    const std::optional<Tensor>& opt_out) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    validate_inputs(elements, test_elements);

    const auto is_in_preprocessing_result = isin_preprocessing(elements, test_elements);

    Tensor output_tensor = ttnn::prim::isin(
        is_in_preprocessing_result.preprocessed_elements_tensor,
        is_in_preprocessing_result.preprocessed_test_elements_tensor,
        is_in_preprocessing_result.single_fetch_elements_number,
        assume_unique,
        invert,
        opt_out);

    return isin_postprocessing(output_tensor, is_in_preprocessing_result);
}

}  // namespace ttnn::operations::experimental
