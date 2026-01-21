// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_abs_pow_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_abs_pow {

std::tuple<uint32_t, float, bool> get_floored_p_and_decimal_and_p_is_negative(float p) {
    auto floored_p = std::floor(p);
    auto decimal = p - floored_p;
    bool p_is_negative = floored_p < 0.0f;
    if (p_is_negative) {
        floored_p = -floored_p;
    }
    return std::make_tuple(static_cast<uint32_t>(floored_p), decimal, p_is_negative);
}

MorehAbsPowOperation::program_factory_t MorehAbsPowOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    // Case for int32
    return MorehAbsPowFactory{};
}

void validate_tensors(
    const MorehAbsPowOperation::operation_attributes_t& /*operation_attributes*/,
    const MorehAbsPowOperation::tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_args.output;

    check_tensor(input, "moreh_abs_pow", "input", {DataType::BFLOAT16, DataType::INT32});
    check_tensor(output, "moreh_abs_pow", "output", {DataType::BFLOAT16, DataType::INT32});
}

void MorehAbsPowOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_tensors(operation_attributes, tensor_args);
};

void MorehAbsPowOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_tensors(operation_attributes, tensor_args);
};
MorehAbsPowOperation::spec_return_value_t MorehAbsPowOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        return tensor_args.output->tensor_spec();
    }
    const auto& input = tensor_args.input;
    return TensorSpec(
        input.logical_shape(),
        TensorLayout(input.dtype(), PageConfig(input.layout()), operation_attributes.memory_config));
}

MorehAbsPowOperation::tensor_return_value_t MorehAbsPowOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        log_debug(tt::LogOp, "{}:{} use output tensor", __func__, __LINE__);
        return {tensor_args.output.value()};
    }

    log_debug(tt::LogOp, "{}:{} create output tensor", __func__, __LINE__);
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
};

}  // namespace ttnn::operations::moreh::moreh_abs_pow

namespace ttnn::prim {
ttnn::operations::moreh::moreh_abs_pow::MorehAbsPowOperation::tensor_return_value_t moreh_abs_pow(
    const Tensor& input,
    const float p,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType = ttnn::operations::moreh::moreh_abs_pow::MorehAbsPowOperation;
    const OperationType::operation_attributes_t operation_attributes{
        p,
        memory_config.value_or(input.memory_config()),
        init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)};
    const OperationType::tensor_args_t tensor_args{input, output};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
