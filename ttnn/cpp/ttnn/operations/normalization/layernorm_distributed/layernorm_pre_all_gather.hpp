// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/layernorm_distributed_types.hpp"
#include "device/layernorm_pre_all_gather_device_operation_types.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

ttnn::Tensor layer_norm_pre_all_gather(
    const ttnn::Tensor& input_tensor,
    DataType dtype = DataType::BFLOAT16,
    const std::optional<const ttnn::Tensor>& residual_input_tensor = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<const ttnn::prim::LayerNormProgramConfig>& program_config = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<const ttnn::Tensor>& recip_tensor = std::nullopt);

}  // namespace ttnn
