// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/embedding_backward/device/embedding_backward_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/constants.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::embedding_backward {

EmbeddingBackwardDeviceOperation::program_factory_t EmbeddingBackwardDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::EmbeddingBackwardProgramFactory{};
}

void EmbeddingBackwardDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void EmbeddingBackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& index_tensor = tensor_args.index_tensor;
    const auto& grad_tensor = tensor_args.grad_tensor;
    const auto& index_tensor_shape = index_tensor.padded_shape();
    const auto& grad_tensor_shape = grad_tensor.padded_shape();

    TT_FATAL(
        index_tensor.layout() == Layout::ROW_MAJOR,
        "Index tensor layout must be ROW_MAJOR but got {}",
        index_tensor.layout());
    TT_FATAL(
        index_tensor.dtype() == DataType::UINT32 or index_tensor.dtype() == DataType::BFLOAT16,
        "Index tensor must be UINT32 or BFLOAT16");

    TT_FATAL(
        index_tensor_shape[1] == 1 && index_tensor_shape[2] == 1,
        "Only dim 0 && 3 for the index tensor can be non 1, but found {} && {}.",
        index_tensor_shape[1],
        index_tensor_shape[2]);

    TT_FATAL(
        index_tensor_shape[-1] % TILE_WIDTH == 0,
        "Number of columns in the index tensor must be divisible by tile width");

    TT_FATAL(
        grad_tensor.layout() == Layout::TILE, "Gradient tensor layout must be TILE but got {}", grad_tensor.layout());
    TT_FATAL(
        grad_tensor.dtype() == DataType::BFLOAT16 or grad_tensor.dtype() == DataType::BFLOAT8_B,
        "Output gradient tensor must be BFLOAT16 or BFLOAT8_B");
    TT_FATAL(
        grad_tensor.dtype() == operation_attributes.output_dtype,
        "Output and input gradient tensors must have the same dtype");

    TT_FATAL(
        grad_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED or
            index_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED or
            operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Embedding b/w does not currently support sharding");

    TT_FATAL(
        grad_tensor_shape[0] == 1 && grad_tensor_shape[1] == 1,
        "First two dimensions for the gradient tensor must be 1, but found {} && {}.",
        grad_tensor_shape[0],
        grad_tensor_shape[1]);

    TT_FATAL(
        grad_tensor_shape[-1] % TILE_WIDTH == 0,
        "Number of columns in the gradient tensor must be divisible by tile width");

    TT_FATAL(
        grad_tensor_shape[2] == index_tensor_shape[0] * index_tensor_shape[-1],
        "Number of rows in gradient tensor must be equal to number of indices in index tensor");
}

EmbeddingBackwardDeviceOperation::spec_return_value_t EmbeddingBackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& grad_tensor = tensor_args.grad_tensor;
    auto embedding_dim = grad_tensor.logical_shape()[-1];

    ttnn::Shape output_shape({1, 1, operation_attributes.num_embeddings, embedding_dim});
    return TensorSpec(
        output_shape,
        TensorLayout(
            operation_attributes.output_dtype, PageConfig(Layout::TILE), operation_attributes.output_mem_config));
}

EmbeddingBackwardDeviceOperation::tensor_return_value_t EmbeddingBackwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.grad_tensor.device());
}

}  // namespace ttnn::operations::embedding_backward

namespace ttnn::prim {
ttnn::Tensor embedding_backward(
    const Tensor& index_tensor,
    const Tensor& grad_tensor,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::DataType& output_dtype,
    uint32_t num_embeddings,
    const std::optional<Tensor>& preallocated_output) {
    using OperationType = ttnn::operations::embedding_backward::EmbeddingBackwardDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .output_mem_config = output_mem_config,
            .output_dtype = output_dtype,
            .num_embeddings = num_embeddings,
        },
        OperationType::tensor_args_t{
            .index_tensor = index_tensor,
            .grad_tensor = grad_tensor,
            .preallocated_output = preallocated_output,
        });
}
}  // namespace ttnn::prim
