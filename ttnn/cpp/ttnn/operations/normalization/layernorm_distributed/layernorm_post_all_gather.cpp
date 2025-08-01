// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_post_all_gather.hpp"

#include "device/layernorm_post_all_gather_op.hpp"

#include "ttnn/operations/normalization/layernorm/device/layernorm_op.hpp"

namespace ttnn::operations::normalization {

ttnn::Tensor ExecuteLayerNormPostAllGather::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& stats,
    float epsilon,
    const std::optional<const ttnn::Tensor>& weight,
    const std::optional<const ttnn::Tensor>& bias,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const LayerNormProgramConfig>& program_config,
    const std::optional<const DataType>& dtype) {
    auto arch = input_tensor.storage_type() == StorageType::DEVICE
                    ? input_tensor.device()->arch()
                    : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();
    auto kernel_config_val =
        init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, true, false, false);
    if (input_tensor.is_sharded()) {
        return tt::tt_metal::operation::run(
                   LayerNorm{
                       .norm_type = LayerNormType::LAYERNORM,
                       .distributed_norm_stage = DistributedLayerNormStage::POST_ALL_GATHER,
                       .eps = epsilon,
                       .output_mem_config = memory_config.value_or(input_tensor.memory_config()),
                       .program_config = program_config.value_or(LayerNormDefaultProgramConfig{}),
                       .compute_kernel_config = kernel_config_val,
                       .dtype = dtype},
                   {input_tensor},
                   {std::nullopt, weight, bias, stats})
            .at(0);
    } else {
        return tt::tt_metal::operation::run(
                   LayerNormPostAllGather{
                       .norm_type = LayerNormDistributedType::LAYERNORM,
                       .eps = epsilon,
                       .memory_config = memory_config.value_or(input_tensor.memory_config()),
                       .compute_kernel_config = kernel_config_val,
                       .dtype = dtype,
                       .use_2d_core_grid = std::nullopt},  // LayerNorm doesn't expose this parameter
                   {input_tensor, stats},
                   {weight, bias})
            .at(0);
    }
}

}  // namespace ttnn::operations::normalization
