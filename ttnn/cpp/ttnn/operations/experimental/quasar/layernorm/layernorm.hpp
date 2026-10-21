// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar {

// Quasar (Metal 2.0) layer_norm. Same contract as ttnn::layer_norm, but only the standard
// (interleaved, non-distributed) Welford path is implemented. The program-config types are shared
// with the original op (ttnn::prim::LayerNormProgramConfig) so callers can pass the same objects.
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
