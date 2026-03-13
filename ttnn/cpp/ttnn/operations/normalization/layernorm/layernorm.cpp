// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm.hpp"
#include <optional>

#include "ttnn/operations/data_movement/clone/clone.hpp"
#include "device/layernorm_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {

DeviceComputeKernelConfig layernorm_default_compute_config(tt::ARCH arch) {
    return init_device_compute_kernel_config(arch, std::nullopt, MathFidelity::HiFi4, false, true);
}

namespace {
// Convert optional<const Tensor> to optional<Tensor> for LayerNormInputs
std::optional<Tensor> to_opt(const std::optional<const Tensor>& t) {
    if (t.has_value()) {
        return t.value();
    }
    return std::nullopt;
}
}  // namespace

Tensor layer_norm(
    const Tensor& input_tensor,
    float epsilon,
    const std::optional<const Tensor>& weight,
    const std::optional<const Tensor>& bias,
    const std::optional<const Tensor>& residual_input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const prim::LayerNormProgramConfig>& program_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const Tensor>& recip_tensor) {
    auto rank = input_tensor.logical_shape().rank();
    TT_FATAL(rank > 0, "LayerNorm operation not supported for 0D tensors. (rank={})", rank);

    if (input_tensor.logical_volume() == 0) [[unlikely]] {
        auto output_memory_config = memory_config.value_or(input_tensor.memory_config());
        return ttnn::clone(input_tensor, /*dtype=*/std::nullopt, output_memory_config, compute_kernel_config);
    }

    auto attrs = prim::LayerNormParams::with_defaults(
        input_tensor, prim::LayerNormType::LAYERNORM, epsilon, memory_config, program_config, compute_kernel_config);

    auto args = prim::LayerNormInputs{
        .input = input_tensor,
        .residual_input_tensor = to_opt(residual_input_tensor),
        .weight = to_opt(weight),
        .bias = to_opt(bias),
        .stats = std::nullopt,
        .recip_tensor = to_opt(recip_tensor)};

    return ttnn::device_operation::launch<prim::LayerNormDeviceOperation>(attrs, args);
}

ttnn::device_operation::OpDescriptorResult<prim::LayerNormDeviceOperation> layer_norm_descriptor(
    const Tensor& input_tensor,
    float epsilon,
    const std::optional<const Tensor>& weight,
    const std::optional<const Tensor>& bias,
    const std::optional<const Tensor>& residual_input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const prim::LayerNormProgramConfig>& program_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const Tensor>& recip_tensor) {
    auto attrs = prim::LayerNormParams::with_defaults(
        input_tensor, prim::LayerNormType::LAYERNORM, epsilon, memory_config, program_config, compute_kernel_config);

    auto args = prim::LayerNormInputs{
        .input = input_tensor,
        .residual_input_tensor = to_opt(residual_input_tensor),
        .weight = to_opt(weight),
        .bias = to_opt(bias),
        .stats = std::nullopt,
        .recip_tensor = to_opt(recip_tensor)};

    return ttnn::device_operation::create_op_descriptor<prim::LayerNormDeviceOperation>(attrs, args);
}

}  // namespace ttnn
