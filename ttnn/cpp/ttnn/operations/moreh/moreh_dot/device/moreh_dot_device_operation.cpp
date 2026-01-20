// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_dot_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_dot {
MorehDotOperation::program_factory_t MorehDotOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    // For now we litteraly don't care and return a single factory. Whatever
    return SingleCore{};
}

void MorehDotOperation::validate(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const auto& input_a = tensor_args.input_a;
    const auto& input_b = tensor_args.input_b;

    TT_FATAL(is_1d_tensor(input_a), "Invalid input tensor dimensions.");
    TT_FATAL(is_1d_tensor(input_b), "Invalid input tensor dimensions.");

    const auto& a_shape_wo_padding = input_a.logical_shape();
    const auto& b_shape_wo_padding = input_b.logical_shape();
    TT_FATAL(a_shape_wo_padding[3] == b_shape_wo_padding[3], "Shape without padding must be the same.");

    TT_FATAL(
        input_a.dtype() == DataType::BFLOAT16 || input_a.dtype() == DataType::BFLOAT8_B, "Unsupported data format");
    TT_FATAL(
        input_a.storage_type() == StorageType::DEVICE and input_b.storage_type() == StorageType::DEVICE,
        "Operands to matmul need to be on device!");
    TT_FATAL(input_a.device() == input_b.device(), "Operands to matmul need to be on the same device!");
    TT_FATAL(
        input_a.buffer() != nullptr and input_b.buffer() != nullptr,
        "Operands to matmul need to be allocated in buffers on device!");
}

void MorehDotOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate(operation_attributes, tensor_args);
}

void MorehDotOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate(operation_attributes, tensor_args);
}

MorehDotOperation::spec_return_value_t MorehDotOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        return tensor_args.output->tensor_spec();
    }
    const auto& input = tensor_args.input_a;
    auto output_shape = input.logical_shape();
    output_shape[3] = 1;
    return TensorSpec(
        output_shape, TensorLayout(input.dtype(), PageConfig(input.layout()), operation_attributes.memory_config));
}

MorehDotOperation::tensor_return_value_t MorehDotOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        return tensor_args.output.value();
    }
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input_a.device());
}

}  // namespace ttnn::operations::moreh::moreh_dot

namespace ttnn::prim {
ttnn::operations::moreh::moreh_dot::MorehDotOperation::tensor_return_value_t moreh_dot(
    const Tensor& input_a,
    const Tensor& input_b,
    const std::optional<Tensor>& output,
    const std::optional<DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType = ttnn::operations::moreh::moreh_dot::MorehDotOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        dtype.value_or(input_a.dtype()),
        memory_config.value_or(input_a.memory_config()),
        init_device_compute_kernel_config(input_a.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)};
    auto tensor_args = OperationType::tensor_args_t{input_a, input_b, output};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
