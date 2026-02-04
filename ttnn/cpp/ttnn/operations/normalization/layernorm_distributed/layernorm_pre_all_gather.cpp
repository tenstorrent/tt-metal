// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_pre_all_gather.hpp"

#include "device/layernorm_pre_all_gather_device_operation.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_device_operation.hpp"
#include "ttnn/device.hpp"

namespace ttnn::operations::normalization {

ttnn::Tensor ExecuteLayerNormPreAllGather::invoke(
    const ttnn::Tensor& input_tensor,
    const DataType dtype,
    const std::optional<const ttnn::Tensor>& residual_input_tensor,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const ttnn::prim::LayerNormProgramConfig>& program_config,
    const std::optional<MemoryConfig>& memory_config) {
    auto arch = input_tensor.storage_type() == StorageType::DEVICE ? input_tensor.device()->arch()
                                                                   : ttnn::GetDefaultDevice()->arch();
    auto kernel_config_val =
        init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, true, false, false);
    if (input_tensor.is_sharded()) {
        return ttnn::prim::layer_norm(
            input_tensor,
            1e-12,                  // epsilon
            std::nullopt,           // weight
            std::nullopt,           // bias
            residual_input_tensor,  // residual_input_tensor
            memory_config.value_or(input_tensor.memory_config()),
            program_config.value_or(ttnn::prim::LayerNormDefaultProgramConfig{}),
            kernel_config_val,
            std::nullopt,  // dtype
            ttnn::prim::LayerNormType::LAYERNORM,
            ttnn::prim::DistributedLayerNormStage::PRE_ALL_GATHER);
    }
    return ttnn::prim::layer_norm_pre_all_gather(
        input_tensor,
        ttnn::prim::LayerNormDistributedType::LAYERNORM,
        dtype,
        kernel_config_val,
        program_config.value_or(ttnn::prim::LayerNormDefaultProgramConfig{}),
        std::nullopt);  // use_2d_core_grid
}

}  // namespace ttnn::operations::normalization
