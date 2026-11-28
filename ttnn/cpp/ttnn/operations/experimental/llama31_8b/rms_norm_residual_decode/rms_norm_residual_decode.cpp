// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/llama31_8b/rms_norm_residual_decode/rms_norm_residual_decode.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/llama31_8b/rms_norm_residual_decode/device/rms_norm_residual_decode_device_operation.hpp"

namespace ttnn::experimental::llama31_8b {

ttnn::Tensor rms_norm_residual_decode(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& residual_tensor,
    const ttnn::Tensor& weight,
    float epsilon,
    const ttnn::prim::LayerNormShardedMultiCoreProgramConfig& program_config,
    const MemoryConfig& memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using namespace ttnn::prim::llama31_8b_rms_norm_residual_decode;
    return ttnn::device_operation::launch<RmsNormResidualDecodeDeviceOperation>(
        RmsNormResidualDecodeParams{
            .eps = epsilon,
            .output_mem_config = memory_config,
            .program_config = program_config,
            .compute_kernel_config = compute_kernel_config},
        RmsNormResidualDecodeInputs{.input = input_tensor, .residual = residual_tensor, .weight = weight});
}

bool rms_norm_residual_decode_supported(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& residual_tensor,
    const ttnn::Tensor& weight,
    const ttnn::prim::LayerNormShardedMultiCoreProgramConfig& program_config,
    const MemoryConfig& memory_config) {
    return ttnn::prim::llama31_8b_rms_norm_residual_decode::in_table(
        input_tensor, residual_tensor, weight, program_config, memory_config);
}

}  // namespace ttnn::experimental::llama31_8b
