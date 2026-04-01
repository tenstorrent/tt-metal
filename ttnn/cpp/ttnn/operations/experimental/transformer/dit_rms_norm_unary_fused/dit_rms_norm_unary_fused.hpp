// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"

namespace ttnn::experimental {

// Fused RMSNorm + unary activation (e.g. SiLU, GELU). Equivalent to:
//   tensor = ttnn.rms_norm(input, ...)
//   ttnn.<activation>(tensor)
// but computed in a single kernel pass.
ttnn::Tensor dit_rms_norm_unary_fused(
    const ttnn::Tensor& input_tensor,
    float epsilon = 1e-5f,
    const std::optional<const ttnn::Tensor>& weight = std::nullopt,
    const std::optional<const ttnn::Tensor>& bias = std::nullopt,
    const std::optional<const ttnn::Tensor>& residual_input_tensor = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<const ttnn::prim::LayerNormProgramConfig>& program_config = std::nullopt,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<ttnn::operations::unary::UnaryWithParam>& activation = std::nullopt);

}  // namespace ttnn::experimental
