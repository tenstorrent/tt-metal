// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_partial_rope.hpp"

#include "device/fused_partial_rope_device_operation.hpp"

namespace ttnn::operations::experimental::transformer::fused_partial_rope {

ttnn::Tensor fused_partial_rope(
    const ttnn::Tensor& input,
    const ttnn::Tensor& cos,
    const ttnn::Tensor& sin,
    const ttnn::Tensor& trans_mat,
    uint32_t rope_dim,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::prim::fused_partial_rope(input, cos, sin, trans_mat, rope_dim, memory_config, compute_kernel_config);
}

}  // namespace ttnn::operations::experimental::transformer::fused_partial_rope
