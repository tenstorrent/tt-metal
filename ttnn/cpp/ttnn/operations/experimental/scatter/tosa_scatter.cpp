// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tosa_scatter.hpp"

#include "device/scatter_device_operation.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/expand/expand.hpp"
#include "ttnn/operations/reduction/reduction_common/reduction_common.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"

namespace ttnn::operations::experimental {
namespace {

constexpr int32_t LAST_DIMENSION = -1;
constexpr int32_t W_DIMENSION = -2;
constexpr int32_t INPUT_RANK_CONSTRAINT = 3;
constexpr int32_t INDEX_RANK_CONSTRAINT = 2;
constexpr int32_t SOURCE_RANK_CONSTRAINT = 3;

enum class InputTensorType : uint8_t { INPUT, INDEX, SOURCE };

namespace CMAKE_UNIQUE_NAMESPACE {

Tensor pre_tosa_scatter_transform_tensor(
    const Tensor& tensor,
    const uint32_t& N,
    const uint32_t& K,
    const uint32_t& W,
    const uint32_t& C,
    const InputTensorType& input_tensor_type) {
    Tensor processed_tensor = tensor;
    if (input_tensor_type == InputTensorType::INDEX) {
        processed_tensor =
            ttnn::expand(ttnn::unsqueeze(tensor, -1), SmallVector<int32_t>{N, W, C}, tensor.memory_config());
        // WARNING: the rest of this if statement is to be removed after fixing the int32 transpose issue (PR: #23415)
        auto device = processed_tensor.device();
        processed_tensor = processed_tensor.cpu();
        processed_tensor = ttnn::to_dtype(processed_tensor, DataType::UINT16);
        processed_tensor = processed_tensor.to_device(device);
    }

    // processed_tensor = expand_tensor(processed_tensor, N, K, W, C, input_tensor_type);
    processed_tensor = ttnn::transpose(processed_tensor, W_DIMENSION, LAST_DIMENSION);
    if (processed_tensor.layout() != Layout::ROW_MAJOR) {
        processed_tensor = ttnn::to_layout(processed_tensor, Layout::ROW_MAJOR);
    }

    return ttnn::unsqueeze_to_4D(processed_tensor);
}

Tensor post_tosa_scatter_transform_tensor(
    Tensor& output_tensor,
    const uint32_t& N,
    const uint32_t& K,
    const uint32_t& W,
    const uint32_t& C,
    const Layout& original_layout) {
    Tensor processed_tensor = ttnn::transpose(output_tensor, W_DIMENSION, LAST_DIMENSION);
    processed_tensor = ttnn::squeeze_from_4D(processed_tensor, INPUT_RANK_CONSTRAINT);
    if (original_layout != Layout::ROW_MAJOR) {
        processed_tensor = ttnn::to_layout(processed_tensor, original_layout);
    }

    return processed_tensor;
}

// input tensors must follow conditions as described at https://www.mlplatform.org/tosa/tosa_spec.html#_scatter
void validate_tensors(const Shape& input_shape, const Shape& index_shape, const Shape& source_shape) {
    TT_FATAL(
        input_shape.rank() == INPUT_RANK_CONSTRAINT,
        "According to TOSA specification, input tensor must be of rank {}, it is {} instead.",
        INPUT_RANK_CONSTRAINT,
        input_shape.rank());

    TT_FATAL(
        index_shape.rank() == INDEX_RANK_CONSTRAINT,
        "According to TOSA specification, index tensor must be of rank {}, it is {} instead.",
        INDEX_RANK_CONSTRAINT,
        input_shape.rank());

    TT_FATAL(
        source_shape.rank() == SOURCE_RANK_CONSTRAINT,
        "According to TOSA specification, source tensor must be of rank {}, it is {} instead.",
        SOURCE_RANK_CONSTRAINT,
        input_shape.rank());

    TT_FATAL(
        input_shape[0] == source_shape[0],
        "Input shape has a different dimension N than source shape (input shape: {}, source shape: {}).",
        input_shape,
        source_shape);

    TT_FATAL(
        input_shape[2] == source_shape[2],
        "Input shape has a different dimension C than source shape (input shape: {}, source shape: {}).",
        input_shape,
        source_shape);

    TT_FATAL(
        input_shape[0] == index_shape[0],
        "Input shape has a different dimension C than index shape (input shape: {}, index shape: {}).",
        input_shape,
        index_shape);

    TT_FATAL(
        index_shape[1] == source_shape[1],
        "Index shape has a different dimension W than source shape (index shape: {}, source shape: {}).",
        index_shape,
        source_shape);
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

Tensor TOSAScatterOperation::invoke(
    const QueueId& queue_id,
    const Tensor& input_tensor,
    const Tensor& index_tensor,
    const Tensor& source_tensor,
    const std::optional<MemoryConfig>& output_memory_config) {
    const auto& input_shape{input_tensor.get_logical_shape()};
    const auto& index_shape{index_tensor.get_logical_shape()};
    const auto& source_shape{source_tensor.get_logical_shape()};

    CMAKE_UNIQUE_NAMESPACE::validate_tensors(input_shape, index_shape, source_shape);

    const uint32_t N = input_shape[0];
    const uint32_t K = input_shape[1];
    const uint32_t W = index_shape[1];
    const uint32_t C = input_shape[2];

    Tensor processed_input_tensor =
        CMAKE_UNIQUE_NAMESPACE::pre_tosa_scatter_transform_tensor(input_tensor, N, K, W, C, InputTensorType::INPUT);

    Tensor processed_index_tensor =
        CMAKE_UNIQUE_NAMESPACE::pre_tosa_scatter_transform_tensor(index_tensor, N, K, W, C, InputTensorType::INDEX);

    Tensor processed_source_tensor =
        CMAKE_UNIQUE_NAMESPACE::pre_tosa_scatter_transform_tensor(source_tensor, N, K, W, C, InputTensorType::SOURCE);

    const MemoryConfig final_memory_config{
        output_memory_config.has_value() ? output_memory_config.value() : input_tensor.memory_config()};

    Tensor output = ttnn::prim::scatter(
        processed_input_tensor,
        LAST_DIMENSION,
        processed_index_tensor,
        processed_source_tensor,
        final_memory_config,
        std::nullopt,
        queue_id);
    return CMAKE_UNIQUE_NAMESPACE::post_tosa_scatter_transform_tensor(output, N, K, W, C, input_tensor.layout());
}

}  // namespace ttnn::operations::experimental
