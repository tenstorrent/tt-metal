// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::llama31_8b {

ttnn::Tensor rms_norm_residual_decode(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& residual_tensor,
    const ttnn::Tensor& weight,
    float epsilon,
    const ttnn::prim::LayerNormShardedMultiCoreProgramConfig& program_config,
    const MemoryConfig& memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config);

bool rms_norm_residual_decode_supported(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& residual_tensor,
    const ttnn::Tensor& weight,
    const ttnn::prim::LayerNormShardedMultiCoreProgramConfig& program_config,
    const MemoryConfig& memory_config);

}  // namespace ttnn::experimental::llama31_8b
