// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim::qsr {

LayerNormDeviceOperation::program_factory_t LayerNormDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return LayerNormWelfordProgramFactory{};
}

void LayerNormDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.input;
    const auto& b = tensor_args.residual_input_tensor;
    const auto& gamma = tensor_args.weight;
    const auto& beta = tensor_args.bias;

    // Scope of this port: interleaved TILE input, standard Welford.
    TT_FATAL(
        std::holds_alternative<ttnn::prim::LayerNormDefaultProgramConfig>(operation_attributes.program_config),
        "Quasar layer_norm only supports LayerNormDefaultProgramConfig (interleaved, non-sharded).");
    TT_FATAL(
        std::get<ttnn::prim::LayerNormDefaultProgramConfig>(operation_attributes.program_config).use_welford,
        "Quasar layer_norm only implements the Welford path; set use_welford=True.");
    TT_FATAL(!a.is_sharded(), "Quasar layer_norm does not support sharded input.");
    TT_FATAL(
        !operation_attributes.output_mem_config.is_sharded(), "Quasar layer_norm does not support sharded output.");
    TT_FATAL(a.layout() == Layout::TILE, "Input tensor must have TILE layout, got: {}", a.layout());
    TT_FATAL(
        a.dtype() == DataType::FLOAT32 || a.dtype() == DataType::BFLOAT16,
        "Input tensor must be FLOAT32 or BFLOAT16 (Quasar has no BFLOAT8_B), got: {}",
        a.dtype());
    TT_FATAL(a.storage_type() == StorageType::DEVICE, "Operands to layernorm need to be on device!");
    TT_FATAL(a.buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");

    // Scope of the current kernels.
    TT_FATAL(!b.has_value(), "Quasar layer_norm does not support residual_input_tensor yet.");
    TT_FATAL(
        a.dtype() != DataType::FLOAT32,
        "Quasar layer_norm: FLOAT32 input is not supported yet (transpose_tile cannot unpack fp32 to DEST, "
        "tt-llk#1559).");

    // Same per-operand contract as the original op, TILE-layout gamma/beta only.
    if (b.has_value()) {
        TT_FATAL(b->layout() == Layout::TILE, "Residual tensor must have TILE layout, got: {}", b->layout());
        TT_FATAL(
            a.logical_shape() == b->logical_shape() && a.padded_shape() == b->padded_shape(),
            "Input and residual shapes must match, got input {} / {} vs residual {} / {}",
            a.logical_shape(),
            a.padded_shape(),
            b->logical_shape(),
            b->padded_shape());
        TT_FATAL(b->buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
        TT_FATAL(a.device() == b->device(), "Input and residual tensors must be on same device");
    }
    for (const auto* affine : {&gamma, &beta}) {
        if (!affine->has_value()) {
            continue;
        }
        const auto& t = affine->value();
        TT_FATAL(t.layout() == Layout::TILE, "Quasar layer_norm requires TILE-layout gamma/beta, got: {}", t.layout());
        TT_FATAL(t.buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
        TT_FATAL(a.device() == t.device(), "Input and gamma/beta tensors must be on same device");
        TT_FATAL(
            t.padded_shape()[-1] == a.padded_shape()[-1],
            "gamma/beta padded width ({}) must match input padded width ({})",
            t.padded_shape()[-1],
            a.padded_shape()[-1]);
    }
    if (gamma.has_value() && beta.has_value()) {
        TT_FATAL(gamma->layout() == beta->layout(), "Gamma and beta must have the same layout!");
    }
}

TensorSpec LayerNormDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    return TensorSpec(
        input.logical_shape(),
        TensorLayout(
            operation_attributes.dtype.value_or(input.dtype()),
            PageConfig(input.layout()),
            operation_attributes.output_mem_config));
}

Tensor LayerNormDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

Tensor layer_norm(
    const Tensor& input_tensor,
    float epsilon,
    const std::optional<const Tensor>& weight,
    const std::optional<const Tensor>& bias,
    const std::optional<const Tensor>& residual_input_tensor,
    const MemoryConfig& output_mem_config,
    const ttnn::prim::LayerNormProgramConfig& program_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const std::optional<const Tensor>& recip_tensor,
    const std::optional<DataType>& dtype) {
    return ttnn::device_operation::launch<LayerNormDeviceOperation>(
        LayerNormParams{
            .eps = epsilon,
            .output_mem_config = output_mem_config,
            .program_config = program_config,
            .compute_kernel_config = compute_kernel_config,
            .dtype = dtype,
        },
        LayerNormInputs{
            .input = input_tensor,
            .residual_input_tensor = residual_input_tensor,
            .weight = weight,
            .bias = bias,
            .recip_tensor = recip_tensor,
        });
}

}  // namespace ttnn::prim::qsr
