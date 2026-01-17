// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_post_all_gather.hpp"

#include "ttnn/operations/normalization/layernorm_distributed/device/layernorm_post_all_gather_device_operation.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_device_operation.hpp"
#include "ttnn/device.hpp"

namespace ttnn::operations::normalization {

ttnn::Tensor ExecuteRMSNormPostAllGather::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& stats,
    float epsilon,
    const std::optional<const ttnn::Tensor>& weight,
    const std::optional<const ttnn::Tensor>& bias,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const ttnn::prim::LayerNormProgramConfig>& program_config,
    const std::optional<const DataType>& dtype,
    const std::optional<bool>& use_2d_core_grid) {
    auto arch = input_tensor.storage_type() == StorageType::DEVICE ? input_tensor.device()->arch()
                                                                   : ttnn::GetDefaultDevice()->arch();
    auto kernel_config_val =
        init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, true, false, false);
    if (input_tensor.is_sharded()) {
        return ttnn::prim::layer_norm(
            input_tensor,
            epsilon,
            weight,
            bias,
            std::nullopt,  // residual_input_tensor
            memory_config.value_or(input_tensor.memory_config()),
            program_config.value_or(ttnn::prim::LayerNormDefaultProgramConfig{}),
            kernel_config_val,
            dtype,
            ttnn::prim::LayerNormType::RMSNORM,
            ttnn::prim::DistributedLayerNormStage::POST_ALL_GATHER,
            stats);
    }
    return ttnn::prim::layer_norm_post_all_gather(
        input_tensor,
        stats,
        ttnn::prim::LayerNormDistributedType::RMSNORM,
        epsilon,
        weight,
        bias,
        memory_config.value_or(input_tensor.memory_config()),
        kernel_config_val,
        dtype,
        use_2d_core_grid,
        program_config.value_or(ttnn::prim::LayerNormDefaultProgramConfig{}));
}

}  // namespace ttnn::operations::normalization
