// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm.hpp"

#include "ttnn/operations/data_movement/clone/clone.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_device_operation.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_common.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/normalization/whole_padding_tiles.hpp"

namespace ttnn {

DeviceComputeKernelConfig rmsnorm_default_compute_config(tt::ARCH arch) {
    bool approx_mode = true;
    bool fp32_acc = false;
    return init_device_compute_kernel_config(
        arch, std::nullopt, tt::tt_metal::MathFidelity::HiFi4, approx_mode, fp32_acc);
}

Tensor rms_norm(
    const Tensor& input_tensor_arg,
    float epsilon,
    const std::optional<const Tensor>& weight_arg,
    const std::optional<const Tensor>& bias_arg,
    const std::optional<const Tensor>& residual_input_tensor_arg,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const prim::LayerNormProgramConfig>& program_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    using operations::normalization::detail::on_default_tile_grid;
    const Tensor input_tensor = on_default_tile_grid(input_tensor_arg);
    const auto on_grid = [](const std::optional<const Tensor>& t) -> std::optional<const Tensor> {
        return t.has_value() ? std::optional<const Tensor>(on_default_tile_grid(*t)) : std::nullopt;
    };
    const auto weight = on_grid(weight_arg);
    const auto bias = on_grid(bias_arg);
    const auto residual_input_tensor = on_grid(residual_input_tensor_arg);

    auto output_memory_config = memory_config.value_or(input_tensor.memory_config());
    auto rank = input_tensor.logical_shape().size();

    TT_FATAL(
        input_tensor.layout() != Layout::ROW_MAJOR,
        "ttnn::rms_norm does not support ROW_MAJOR input tensors. Use TILE layout.");

    // For 0V tensors
    if (input_tensor.logical_volume() == 0) [[unlikely]] {
        return ttnn::clone(input_tensor, /*dtype=*/std::nullopt, output_memory_config, compute_kernel_config);
    }

    // For 0D tensors
    if (rank == 0) [[unlikely]] {
        auto result = ttnn::multiply(
            input_tensor,
            ttnn::rsqrt(
                ttnn::add(
                    ttnn::square(input_tensor, output_memory_config),
                    epsilon,
                    /*output_dtype=*/std::nullopt,
                    output_memory_config),
                /*fast_and_approximate_mode=*/false,
                output_memory_config),
            /*output_dtype=*/std::nullopt,
            output_memory_config);

        if (weight.has_value()) {
            result = ttnn::multiply(result, weight.value(), /*output_dtype=*/std::nullopt, output_memory_config);
        }
        if (bias.has_value()) {
            result = ttnn::add(result, bias.value(), /*output_dtype=*/std::nullopt, output_memory_config);
        }
        return result;
    }

    auto arch = input_tensor.storage_type() == StorageType::DEVICE ? input_tensor.device()->arch()
                                                                   : ttnn::GetDefaultDevice()->arch();
    ttnn::verify_numerical_configuration(arch, compute_kernel_config);
    auto kernel_config_val = compute_kernel_config.value_or(rmsnorm_default_compute_config(arch));
    return ttnn::prim::layer_norm(
        input_tensor,
        epsilon,
        weight,
        bias,
        residual_input_tensor,
        output_memory_config,
        program_config.value_or(ttnn::prim::create_layernorm_program_config(
            input_tensor.shard_spec(),
            input_tensor.tensor_spec().tile().get_height(),
            input_tensor.tensor_spec().tile().get_width())),
        kernel_config_val,
        std::nullopt,  // dtype
        prim::LayerNormType::RMSNORM);
}

}  // namespace ttnn
