// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "full_like_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <optional>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

FullLikeDeviceOperation::program_factory_t FullLikeDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    if (operation_attributes.memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED) {
        return FullLikeInterleavedProgramFactory{};
    }
    return FullLikeNDShardedProgramFactory{};
}

void FullLikeDeviceOperation::validate(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    // Currently ther program factory only supports these 3 datatypes:
    TT_FATAL(
        operation_attributes.dtype == DataType::BFLOAT16 || operation_attributes.dtype == DataType::INT32 ||
            operation_attributes.dtype == DataType::FLOAT32,
        "Full Like: Unsupported data type {}",
        operation_attributes.dtype);
    if (operation_attributes.dtype != input.dtype()) {
        TT_FATAL(input.layout() == Layout::TILE, "Full Like: Data type conversion is only supported with tile layout");
    }
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Full Like: Input must be on device");
    TT_FATAL(input.buffer() != nullptr, "Full Like: Input must be allocated in buffer on device");
    // TT_FATAL(
    //     input.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
    //     "Full Like: Not currently supporting sharding");
    // TT_FATAL(
    //     operation_attributes.memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
    //     "Full Like: Not currently supporting sharding");
}

void FullLikeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate(operation_attributes, tensor_args);
}

FullLikeDeviceOperation::spec_return_value_t FullLikeDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return TensorSpec(
        tensor_args.input.logical_shape(),
        tt::tt_metal::TensorLayout(
            operation_attributes.dtype,
            tt::tt_metal::PageConfig(operation_attributes.layout),
            operation_attributes.memory_config));
}

FullLikeDeviceOperation::tensor_return_value_t FullLikeDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input.device());
}

tensor_return_value_t moreh_full_like(
    const Tensor& input,
    std::variant<float, int> fill_value,
    const std::optional<DataType>& dtype,
    const std::optional<Layout>& layout,
    const std::optional<MemoryConfig>& memory_config) {
    using OperationType = FullLikeDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        fill_value,
        dtype.value_or(input.dtype()),
        layout.value_or(input.layout()),
        memory_config.value_or(input.memory_config())};
    auto tensor_args = OperationType::tensor_args_t{input};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
