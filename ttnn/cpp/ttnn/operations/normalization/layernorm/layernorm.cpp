// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm.hpp"
#include <optional>

#include "ttnn/operations/data_movement/clone/clone.hpp"
#include "device/layernorm_device_operation.hpp"
#include "device/layernorm_common.hpp"
#include "ttnn/device.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn {

DeviceComputeKernelConfig layernorm_default_compute_config(tt::ARCH arch) {
    bool approx_mode = false;
    bool fp32_acc = true;
    return init_device_compute_kernel_config(arch, std::nullopt, MathFidelity::HiFi4, approx_mode, fp32_acc);
}

std::pair<prim::LayerNormParams, prim::LayerNormInputs> prepare_norm(
    const Tensor& input_tensor,
    float epsilon,
    prim::LayerNormType norm_type,
    const DeviceComputeKernelConfig& default_compute_config,
    const std::optional<const Tensor>& weight,
    const std::optional<const Tensor>& bias,
    const std::optional<const Tensor>& residual_input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const prim::LayerNormProgramConfig>& program_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const Tensor>& recip_tensor) {
    auto output_memory_config = memory_config.value_or(input_tensor.memory_config());

    auto kernel_config_val = compute_kernel_config.value_or(default_compute_config);

    auto prog_config = program_config.value_or(ttnn::prim::create_layernorm_program_config(
        input_tensor.shard_spec(),
        input_tensor.tensor_spec().tile().get_height(),
        input_tensor.tensor_spec().tile().get_width()));

    auto attrs = prim::LayerNormParams{
        .norm_type = norm_type,
        .distributed_norm_stage = prim::DistributedLayerNormStage::NOT_DISTRIBUTED,
        .eps = epsilon,
        .output_mem_config = output_memory_config,
        .program_config = prog_config,
        .compute_kernel_config = kernel_config_val};

    // Convert optional<const Tensor> to optional<Tensor> for LayerNormInputs
    auto to_opt = [](const std::optional<const Tensor>& t) -> std::optional<Tensor> {
        if (t.has_value()) {
            return t.value();
        }
        return std::nullopt;
    };

    auto args = prim::LayerNormInputs{
        .input = input_tensor,
        .residual_input_tensor = to_opt(residual_input_tensor),
        .weight = to_opt(weight),
        .bias = to_opt(bias),
        .stats = std::nullopt,
        .recip_tensor = to_opt(recip_tensor)};

    return {std::move(attrs), std::move(args)};
}

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

    // For 0D tensors
    TT_FATAL(rank > 0, "LayerNorm operation not supported for 0D tensors. (rank={})", rank);

    // For 0V tensors
    auto output_memory_config = memory_config.value_or(input_tensor.memory_config());
    if (input_tensor.logical_volume() == 0) [[unlikely]] {
        return ttnn::clone(input_tensor, /*dtype=*/std::nullopt, output_memory_config, compute_kernel_config);
    }

    auto arch = input_tensor.storage_type() == StorageType::DEVICE ? input_tensor.device()->arch()
                                                                   : ttnn::GetDefaultDevice()->arch();
    auto [attrs, args] = prepare_norm(
        input_tensor,
        epsilon,
        prim::LayerNormType::LAYERNORM,
        layernorm_default_compute_config(arch),
        weight,
        bias,
        residual_input_tensor,
        memory_config,
        program_config,
        compute_kernel_config,
        recip_tensor);

    return ttnn::prim::layer_norm(
        input_tensor,
        epsilon,
        weight,
        bias,
        residual_input_tensor,
        attrs.output_mem_config,
        attrs.program_config,
        attrs.compute_kernel_config,
        std::nullopt,                                      // dtype
        prim::LayerNormType::LAYERNORM,                    // norm_type
        prim::DistributedLayerNormStage::NOT_DISTRIBUTED,  // distributed_norm_stage
        std::nullopt,                                      // stats
        recip_tensor);
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
    auto arch = input_tensor.storage_type() == StorageType::DEVICE ? input_tensor.device()->arch()
                                                                   : ttnn::GetDefaultDevice()->arch();
    auto [attrs, args] = prepare_norm(
        input_tensor,
        epsilon,
        prim::LayerNormType::LAYERNORM,
        layernorm_default_compute_config(arch),
        weight,
        bias,
        residual_input_tensor,
        memory_config,
        program_config,
        compute_kernel_config,
        recip_tensor);

    return ttnn::device_operation::create_op_descriptor<prim::LayerNormDeviceOperation>(attrs, args);
}

}  // namespace ttnn
