// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_layer_norm_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::moreh::moreh_layer_norm {

namespace {
inline void check_tensor(const Tensor& tensor, const std::string& op_name) {
    TT_FATAL(tensor.layout() == Layout::TILE, "{} only supports tiled layout. Got: {}", op_name, tensor.layout());
    TT_FATAL(tensor.dtype() == DataType::BFLOAT16, "{} only supports bfloat16. Got: {}", op_name, tensor.dtype());
    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE,
        "Operands to {} need to be on device! Got: {}",
        op_name,
        tensor.storage_type());
    TT_FATAL(tensor.buffer() != nullptr, "Operands to {} need to be allocated in buffers on device!", op_name);
}
}  // namespace

void MorehLayerNormOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    const auto& gamma = tensor_args.gamma;
    const auto& beta = tensor_args.beta;

    check_tensor(input, "moreh_layer_norm");

    const auto normalized_dims = operation_attributes.normalized_dims;

    TT_FATAL(normalized_dims > 0, "normalized_dims should > 0. Got {}", normalized_dims);
    TT_FATAL(
        normalized_dims <= input.padded_shape().rank(),
        "normalized_dims should <= input rank ({}). Got: {}",
        input.padded_shape().rank(),
        normalized_dims);

    if (gamma.has_value()) {
        check_tensor(gamma.value(), "moreh_layer_norm");
        TT_FATAL(input.device() == gamma.value().device(), "input and gamma should be on the same device.");
    }

    if (beta.has_value()) {
        check_tensor(beta.value(), "moreh_layer_norm");
        TT_FATAL(input.device() == beta.value().device(), "input and beta should be on the same device");
    }

    const auto& mean = tensor_args.mean;
    const auto& rstd = tensor_args.rstd;

    if (mean.has_value()) {
        check_tensor(mean.value(), "moreh_layer_norm");
    }

    if (rstd.has_value()) {
        check_tensor(rstd.value(), "moreh_layer_norm");
    }
}

MorehLayerNormOperation::program_factory_t MorehLayerNormOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return ProgramFactory{};
}

void MorehLayerNormOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void MorehLayerNormOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

MorehLayerNormOperation::spec_return_value_t MorehLayerNormOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input = tensor_args.input;
    std::vector<std::optional<TensorSpec>> result(3);

    if (tensor_args.output.has_value()) {
        result[0] = tensor_args.output->tensor_spec();
    } else {
        result[0] = TensorSpec(
            input.logical_shape(),
            TensorLayout(input.dtype(), PageConfig(Layout::TILE), operation_attributes.memory_config));
    }

    if (tensor_args.mean.has_value()) {
        result[1] = tensor_args.mean->tensor_spec();
    }

    if (tensor_args.rstd.has_value()) {
        result[2] = tensor_args.rstd->tensor_spec();
    }

    return result;
}

MorehLayerNormOperation::tensor_return_value_t MorehLayerNormOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_specs = compute_output_specs(operation_attributes, tensor_args);

    std::vector<std::optional<Tensor>> result(3);

    if (tensor_args.output.has_value()) {
        result[0] = tensor_args.output.value();
    } else {
        result[0] = create_device_tensor(*output_specs.at(0), tensor_args.input.device());
    }

    if (tensor_args.mean.has_value()) {
        result[1] = tensor_args.mean.value();
    }

    if (tensor_args.rstd.has_value()) {
        result[2] = tensor_args.rstd.value();
    }

    return result;
}

}  // namespace ttnn::operations::moreh::moreh_layer_norm

namespace ttnn::prim {
ttnn::operations::moreh::moreh_layer_norm::MorehLayerNormOperation::tensor_return_value_t moreh_layer_norm(
    const Tensor& input,
    uint32_t normalized_dims,
    float eps,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    const std::optional<const Tensor>& output,
    const std::optional<const Tensor>& mean,
    const std::optional<const Tensor>& rstd,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType = ttnn::operations::moreh::moreh_layer_norm::MorehLayerNormOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        normalized_dims,
        eps,
        memory_config.value_or(input.memory_config()),
        init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)};
    auto tensor_args = OperationType::tensor_args_t{input, gamma, beta, output, mean, rstd};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
