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

namespace ttnn {

DeviceComputeKernelConfig rmsnorm_default_compute_config(tt::ARCH arch) {
    bool approx_mode = true;
    bool fp32_acc = false;
    return init_device_compute_kernel_config(arch, std::nullopt, tt::tt_metal::MathFidelity::HiFi4, approx_mode, fp32_acc);
}

Tensor rms_norm(
    const Tensor& input_tensor,
    float epsilon,
    const std::optional<const Tensor>& weight,
    const std::optional<const Tensor>& bias,
    const std::optional<const Tensor>& residual_input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const prim::LayerNormProgramConfig>& program_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<DataType>& dtype_arg,
    const std::optional<Tensor>& output_tensor,
    uint32_t output_col_tiles,
    const std::vector<Tensor>& extra_output_tensors) {
    const std::optional<DataType> dtype =
        output_tensor.has_value() ? std::optional<DataType>(output_tensor.value().dtype()) : dtype_arg;
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
    auto kernel_config_val = compute_kernel_config.value_or(rmsnorm_default_compute_config(arch));
    // fp32 math to a bf16 result: round in DEST to nearest even, as ttnn.typecast does (the packer's own fp32 ->
    // bf16 conversion rounds some values otherwise), so the result equals typecast(rms_norm(x)).
    std::optional<operations::unary::UnaryWithParam> round_bf16;
    const bool fp32_dest = std::get<2>(get_compute_kernel_config_args(arch, kernel_config_val));
    if (dtype == DataType::BFLOAT16 && input_tensor.dtype() == DataType::FLOAT32 && fp32_dest) {
        round_bf16 = operations::unary::UnaryWithParam(
            operations::unary::UnaryOpType::TYPECAST,
            {static_cast<float>(DataType::FLOAT32), static_cast<float>(DataType::BFLOAT16)});
    }
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
        dtype,
        prim::LayerNormType::RMSNORM,
        prim::DistributedLayerNormStage::NOT_DISTRIBUTED,
        std::nullopt,  // stats
        std::nullopt,  // recip_tensor
        round_bf16,
        output_tensor,
        output_col_tiles,
        extra_output_tensors);
}

}  // namespace ttnn
