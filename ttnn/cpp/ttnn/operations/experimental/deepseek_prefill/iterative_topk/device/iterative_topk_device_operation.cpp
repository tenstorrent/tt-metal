// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "iterative_topk_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::iterative_topk {

void IterativeTopkDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    TT_FATAL(input.storage_type() == tt::tt_metal::StorageType::DEVICE, "Input tensor must be on device");
    TT_FATAL(input.buffer() != nullptr, "Input tensor must be allocated");
    TT_FATAL(input.dtype() == tt::tt_metal::DataType::FLOAT32, "Input tensor must be FLOAT32");
    TT_FATAL(input.layout() == tt::tt_metal::Layout::ROW_MAJOR, "Input tensor must be ROW_MAJOR layout");
    TT_FATAL(
        input.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "Input tensor must be interleaved");
    TT_FATAL(attributes.k > 0, "k must be > 0");
    TT_FATAL(
        attributes.k <= input.logical_shape()[-1],
        "k ({}) must be <= last dimension size ({})",
        attributes.k,
        input.logical_shape()[-1]);
    TT_FATAL(input.logical_shape().rank() >= 2, "Input tensor must be at least 2D");
}

IterativeTopkDeviceOperation::spec_return_value_t IterativeTopkDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    auto shape = input.logical_shape();
    auto output_shape = shape;
    output_shape[-1] = attributes.k;

    return std::array<TensorSpec, 2>{
        TensorSpec(
            output_shape,
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::FLOAT32,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                attributes.output_mem_config)),
        TensorSpec(
            output_shape,
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::UINT32,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                attributes.output_mem_config))};
}

IterativeTopkDeviceOperation::tensor_return_value_t IterativeTopkDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    auto specs = compute_output_specs(attributes, tensor_args);
    return std::array<Tensor, 2>{
        create_device_tensor(specs[0], tensor_args.input.device()),
        create_device_tensor(specs[1], tensor_args.input.device())};
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::iterative_topk

namespace ttnn::prim {

ttnn::operations::experimental::deepseek_prefill::iterative_topk::IterativeTopkDeviceOperation::tensor_return_value_t
iterative_topk(const Tensor& input, uint32_t k, const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config) {
    using OperationType =
        ttnn::operations::experimental::deepseek_prefill::iterative_topk::IterativeTopkDeviceOperation;

    auto operation_attributes =
        OperationType::operation_attributes_t{k, output_mem_config.value_or(input.memory_config())};
    auto tensor_args = OperationType::tensor_args_t{input};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
