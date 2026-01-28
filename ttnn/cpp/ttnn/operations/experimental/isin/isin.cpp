// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "isin.hpp"

#include "isin_common.hpp"
#include "device/isin_device_operation.hpp"

#include "tt-metalium/shape.hpp"
#include "ttnn/operations/core/core.hpp"

#include <enchantum/enchantum.hpp>

namespace ttnn::operations::experimental {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
struct IsInPreprocessingResult {
    Tensor preprocessed_elements_tensor;
    Tensor preprocessed_test_elements_tensor;
    Shape original_elements_shape{};
    Layout original_elements_layout{};
    uint32_t elements_size{};
    uint32_t test_elements_size{};
    uint32_t single_fetch_elements_number{};
};

void validate_inputs(const Tensor& elements, const Tensor& test_elements) {
    // Ensure both input tensors have the same data type for element comparison
    TT_FATAL(
        elements.dtype() == test_elements.dtype(),
        "elements.dtype: {}, test_elements.dtype: {}, they must be equal",
        enchantum::to_string(elements.dtype()),
        enchantum::to_string(test_elements.dtype()));
}

uint32_t calculate_max_fetch_size(const Tensor& elements, const Tensor& test_elements) {
    // Get available L1 memory per core
    const auto l1_size_per_core = elements.device()->l1_size_per_core();
    // Apply 80% margin for safety to avoid memory overflow
    const auto confidence_margin = 0.8f;
    // Calculate size of individual elements in bytes
    const auto elements_datum_size = elements.element_size();
    const auto test_elements_datum_size = test_elements.element_size();
    const auto output_datum_size = elements_datum_size;

    // Calculate max elements that fit in L1 considering all three buffers (elements, test_elements, output)
    // Round down to nearest multiple of 64 for alignment
    return (static_cast<uint32_t>(
               l1_size_per_core * confidence_margin /
               (elements_datum_size + test_elements_datum_size + output_datum_size))) &
           static_cast<uint32_t>(~(63U));
}

IsInPreprocessingResult preprocess_isin_inputs(const Tensor& elements, const Tensor& test_elements) {
    IsInPreprocessingResult is_in_preprocessing_result;
    // Initialize with input tensors
    is_in_preprocessing_result.preprocessed_elements_tensor = elements;
    is_in_preprocessing_result.preprocessed_test_elements_tensor = test_elements;
    // Store original shape and layout for restoration in postprocessing
    is_in_preprocessing_result.original_elements_shape =
        is_in_preprocessing_result.preprocessed_elements_tensor.logical_shape();
    is_in_preprocessing_result.original_elements_layout =
        is_in_preprocessing_result.preprocessed_elements_tensor.layout();
    // Calculate total number of elements in each tensor
    is_in_preprocessing_result.elements_size = is_in_preprocessing_result.preprocessed_elements_tensor.logical_volume();
    is_in_preprocessing_result.test_elements_size =
        is_in_preprocessing_result.preprocessed_test_elements_tensor.logical_volume();
    // Determine optimal batch size based on L1 memory constraints
    is_in_preprocessing_result.single_fetch_elements_number = calculate_max_fetch_size(
        is_in_preprocessing_result.preprocessed_elements_tensor,
        is_in_preprocessing_result.preprocessed_test_elements_tensor);

    // Convert elements tensor to row-major layout if needed for efficient processing
    if (is_in_preprocessing_result.preprocessed_elements_tensor.layout() !=
        ttnn::experimental::prim::OUTPUT_TENSOR_LAYOUT) {
        is_in_preprocessing_result.preprocessed_elements_tensor = ttnn::to_layout(
            is_in_preprocessing_result.preprocessed_elements_tensor, ttnn::experimental::prim::OUTPUT_TENSOR_LAYOUT);
    }
    // Convert test_elements tensor to row-major layout if needed
    if (is_in_preprocessing_result.preprocessed_test_elements_tensor.layout() != Layout::ROW_MAJOR) {
        is_in_preprocessing_result.preprocessed_test_elements_tensor =
            ttnn::to_layout(is_in_preprocessing_result.preprocessed_test_elements_tensor, Layout::ROW_MAJOR);
    }

    // Flatten elements tensor to 1D if it has multiple dimensions
    if (elements.logical_shape().rank() > ttnn::experimental::prim::OUTPUT_TENSOR_RANK) {
        is_in_preprocessing_result.preprocessed_elements_tensor = ttnn::reshape(
            is_in_preprocessing_result.preprocessed_elements_tensor, Shape{is_in_preprocessing_result.elements_size});
    }
    // Flatten test_elements tensor to 1D if needed
    if (test_elements.logical_shape().rank() > ttnn::experimental::prim::OUTPUT_TENSOR_RANK) {
        is_in_preprocessing_result.preprocessed_test_elements_tensor = ttnn::reshape(
            is_in_preprocessing_result.preprocessed_test_elements_tensor,
            Shape{is_in_preprocessing_result.test_elements_size});
    }

    return is_in_preprocessing_result;
}

Tensor process_isin_output(Tensor& output_tensor, const IsInPreprocessingResult& is_in_preprocessing_result) {
    // Restore original shape if it was flattened during preprocessing
    if (is_in_preprocessing_result.original_elements_shape.rank() != ttnn::experimental::prim::OUTPUT_TENSOR_RANK) {
        output_tensor = ttnn::reshape(output_tensor, is_in_preprocessing_result.original_elements_shape);
    }
    // Restore original layout if it was changed during preprocessing
    if (is_in_preprocessing_result.original_elements_layout != ttnn::experimental::prim::OUTPUT_TENSOR_LAYOUT) {
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

    // Validate that input tensors have compatible data types
    validate_inputs(elements, test_elements);

    // Prepare tensors: convert to row-major layout, flatten to 1D, and calculate batch size
    const auto is_in_preprocessing_result = preprocess_isin_inputs(elements, test_elements);

    // Execute the device operation to check which elements are in test_elements
    Tensor output_tensor = ttnn::experimental::prim::isin(
        is_in_preprocessing_result.preprocessed_elements_tensor,
        is_in_preprocessing_result.preprocessed_test_elements_tensor,
        is_in_preprocessing_result.single_fetch_elements_number,
        assume_unique,
        invert,
        opt_out);

    // Restore output tensor to original shape and layout
    return process_isin_output(output_tensor, is_in_preprocessing_result);
}

}  // namespace ttnn::operations::experimental
