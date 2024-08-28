// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_pre_all_gather.hpp"

#include "ttnn/operations/normalization/layernorm/device/layernorm_op.hpp"

namespace ttnn::operations::normalization {

ttnn::Tensor ExecuteLayerNormPreAllGather::invoke(
    const ttnn::Tensor& input_tensor,
    const std::optional<const LayerNormProgramConfig>& program_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    auto arch = input_tensor.storage_type() == StorageType::DEVICE ? input_tensor.device()->arch() : AutoFormat::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, true, false, false);
    return operation::run(
                LayerNorm{
                    .norm_type = LayerNormType::LAYERNORM,
                    .distributed_type = LayerNormDistributedType::PRE_ALL_GATHER,
                    .eps = 1e-12,
                    .program_config = program_config.value_or(LayerNormDefaultProgramConfig{}),
                    .compute_kernel_config = kernel_config_val},
                {input_tensor}).at(0);
}

}  // namespace ttnn::operations::normalization
