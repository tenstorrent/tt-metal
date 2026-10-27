// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation_types.hpp"

namespace ttnn::operations::experimental::minimal_matmul {

// Re-export the config type for backward compatibility
using MinimalMatmulConfig = ttnn::experimental::prim::MinimalMatmulConfig;

}  // namespace ttnn::operations::experimental::minimal_matmul

namespace ttnn::experimental {

// input_tensor: a single activation, or a 2-element list [prefix, suffix] virtually concatenated
// (concat-free) on the channel (K, last) axis ONLY — the two must be identical on every other axis.
// Any per-segment channel count is allowed; segments are joined at tile boundaries with zero-filled
// gaps on both the activation (zero from tilization) and the weight (the stacked weight must be
// per-segment tile-padded, i.e. zero rows between segments, via prepare_weight_for_concatenated_input).
// The op size-checks prefix_padded_K + suffix_padded_K == weight_padded_K but trusts the weight
// layout for gap content.
ttnn::Tensor minimal_matmul(
    const std::variant<ttnn::Tensor, std::vector<ttnn::Tensor>>& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<ttnn::Tensor>& bias_tensor,
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation,
    const std::optional<const ttnn::experimental::prim::MinimalMatmulConfig>& config,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<const DataType> dtype = std::nullopt,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    bool fuse_swiglu = false);
}
