// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/quasar/layer_norm/device/layernorm_types_qsr.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar {

DeviceComputeKernelConfig layernorm_default_compute_config(tt::ARCH arch);

Tensor layer_norm(
    const Tensor& input_tensor,
    float epsilon = 1e-12,
    const std::optional<const Tensor>& weight = std::nullopt,
    const std::optional<const Tensor>& bias = std::nullopt,
    const std::optional<const Tensor>& residual_input_tensor = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<const ttnn::prim::LayerNormProgramConfig>& program_config = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<const Tensor>& recip_tensor = std::nullopt);

}  // namespace ttnn::operations::experimental::quasar
