// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_falcon7b_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/work_split.hpp>
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

NlpCreateHeadsFalcon7BDeviceOperation::program_factory_t NlpCreateHeadsFalcon7BDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return NlpCreateQkvHeadsFalcon7BProgramFactory{};
}

void NlpCreateHeadsFalcon7BDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args;
    const auto& input_shape = input_tensor.padded_shape();

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor.dtype() == tt::tt_metal::DataType::FLOAT32 ||
            input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16 ||
            input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format");
    TT_FATAL(
        input_tensor.layout() == Layout::TILE, "Input tensor layout must be TILE but got {}", input_tensor.layout());

    TT_FATAL(
        input_shape[2] % tt::constants::TILE_HEIGHT == 0,
        "Input shape[2] ({}) must be divisible by TILE_HEIGHT ({})",
        input_shape[2],
        tt::constants::TILE_HEIGHT);
    TT_FATAL((input_shape == ttnn::Shape({input_shape[0], 1, input_shape[2], 4672})), "Unsupported input shape");
    TT_FATAL(
        operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Output memory config layout must be INTERLEAVED but got {}",
        operation_attributes.output_mem_config.memory_layout());
}

void NlpCreateHeadsFalcon7BDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

NlpCreateHeadsFalcon7BDeviceOperation::spec_return_value_t NlpCreateHeadsFalcon7BDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (operation_attributes.output_mem_config.is_sharded()) {
        TT_FATAL(false, "Sharded output memory config is not supported for nlp_create_qkv_heads_falcon7b");
    }

    const auto& input_tensor = tensor_args;
    const auto& input_shape = input_tensor.padded_shape();
    tt::tt_metal::TensorLayout layout(
        input_tensor.dtype(), tt::tt_metal::PageConfig(Layout::TILE), operation_attributes.output_mem_config);
    return {
        .q = TensorSpec(Shape({input_shape[0], 71, input_shape[2], 64}), layout),
        .k = TensorSpec(Shape({input_shape[0], 1, input_shape[2], 64}), layout),
        .v = TensorSpec(Shape({input_shape[0], 1, input_shape[2], 64}), layout)};
}

NlpCreateHeadsFalcon7BDeviceOperation::tensor_return_value_t
NlpCreateHeadsFalcon7BDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_specs = compute_output_specs(operation_attributes, tensor_args);
    auto* device = tensor_args.device();

    return {
        .q = create_device_tensor(output_specs.q, device),
        .k = create_device_tensor(output_specs.k, device),
        .v = create_device_tensor(output_specs.v, device)};
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::experimental::prim::NlpCreateQkvHeadsFalcon7bResult nlp_create_qkv_heads_falcon7b(
    const Tensor& input, const std::optional<tt::tt_metal::MemoryConfig>& memory_config) {
    using OperationType = ttnn::experimental::prim::NlpCreateHeadsFalcon7BDeviceOperation;

    const tt::tt_metal::MemoryConfig output_mem_config = memory_config.value_or(input.memory_config());
    auto operation_attributes = OperationType::operation_attributes_t{output_mem_config};
    const auto& tensor_args = input;

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
