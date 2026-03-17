// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm.hpp"

#include "ttnn/operations/data_movement/clone/clone.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn {

DeviceComputeKernelConfig rmsnorm_default_compute_config(tt::ARCH arch) {
    return init_device_compute_kernel_config(arch, std::nullopt, MathFidelity::HiFi4, true, false);
}

namespace {
std::optional<Tensor> to_opt(const std::optional<const Tensor>& t) {
    if (t.has_value()) {
        return t.value();
    }
    return std::nullopt;
}
}  // namespace

Tensor rms_norm(
    const Tensor& input_tensor,
    float epsilon,
    const std::optional<const Tensor>& weight,
    const std::optional<const Tensor>& bias,
    const std::optional<const Tensor>& residual_input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const prim::LayerNormProgramConfig>& program_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    auto output_memory_config = memory_config.value_or(input_tensor.memory_config());
    auto rank = input_tensor.logical_shape().size();

    TT_FATAL(
        input_tensor.layout() != Layout::ROW_MAJOR,
        "ttnn::rms_norm does not support ROW_MAJOR input tensors. Use TILE layout.");

    if (input_tensor.logical_volume() == 0) [[unlikely]] {
        return ttnn::clone(input_tensor, /*dtype=*/std::nullopt, output_memory_config, compute_kernel_config);
    }

    if (rank == 0) [[unlikely]] {
        auto result = ttnn::divide(
            input_tensor, ttnn::abs(input_tensor, output_memory_config), /*alpha=*/std::nullopt, output_memory_config);

        if (weight.has_value()) {
            result = ttnn::multiply(result, weight.value(), /*alpha=*/std::nullopt, output_memory_config);
        }
        if (bias.has_value()) {
            result = ttnn::add(result, bias.value(), /*alpha=*/std::nullopt, output_memory_config);
        }
        return result;
    }

    auto attrs = prim::LayerNormParams::with_defaults(
        input_tensor, prim::LayerNormType::RMSNORM, epsilon, memory_config, program_config, compute_kernel_config);

    auto args = prim::LayerNormInputs{
        .input = input_tensor,
        .residual_input_tensor = to_opt(residual_input_tensor),
        .weight = to_opt(weight),
        .bias = to_opt(bias),
        .stats = std::nullopt,
        .recip_tensor = std::nullopt};

    return ttnn::device_operation::launch<prim::LayerNormDeviceOperation>(attrs, args);
}

ttnn::device_operation::OpDescriptorResult<prim::LayerNormDeviceOperation> rms_norm_descriptor(
    const Tensor& input_tensor,
    float epsilon,
    const std::optional<const Tensor>& weight,
    const std::optional<const Tensor>& bias,
    const std::optional<const Tensor>& residual_input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const prim::LayerNormProgramConfig>& program_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    auto attrs = prim::LayerNormParams::with_defaults(
        input_tensor, prim::LayerNormType::RMSNORM, epsilon, memory_config, program_config, compute_kernel_config);

    auto args = prim::LayerNormInputs{
        .input = input_tensor,
        .residual_input_tensor = to_opt(residual_input_tensor),
        .weight = to_opt(weight),
        .bias = to_opt(bias),
        .stats = std::nullopt,
        .recip_tensor = std::nullopt};

    return ttnn::device_operation::create_op_descriptor<prim::LayerNormDeviceOperation>(attrs, args);
}

}  // namespace ttnn
