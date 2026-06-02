// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "dit_rms_norm_unary_fused.hpp"

namespace ttnn::experimental {

// TODO(nuked-op layer_norm): this op fused RMSNorm + activation by calling the
// nuked ttnn::prim::layer_norm primitive. Restore the real fused call once the
// layer_norm op is recreated. Passthrough stub below only satisfies the type
// system so the build compiles.
ttnn::Tensor dit_rms_norm_unary_fused(
    const ttnn::Tensor& input_tensor,
    float /*epsilon*/,
    const std::optional<const ttnn::Tensor>& /*weight*/,
    const std::optional<const ttnn::Tensor>& /*bias*/,
    const std::optional<const ttnn::Tensor>& /*residual_input_tensor*/,
    const std::optional<MemoryConfig>& /*memory_config*/,
    const std::optional<const ttnn::prim::LayerNormProgramConfig>& /*program_config*/,
    std::optional<const ttnn::DeviceComputeKernelConfig> /*compute_kernel_config*/,
    const std::optional<ttnn::operations::unary::UnaryWithParam>& /*activation*/) {
    // TODO(nuked-op layer_norm): restore real ttnn::prim::layer_norm call.
    return input_tensor;
}

}  // namespace ttnn::experimental
