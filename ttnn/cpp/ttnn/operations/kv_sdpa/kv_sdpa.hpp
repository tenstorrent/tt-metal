// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {

// Specialized fused-flash scaled-dot-product attention for the small-query MQA case (Q length == one
// tile, single/grouped KV head, non-causal full attention). Runs the production transformer-SDPA
// online-softmax (sdpa_standard) with one core per Q head, specialized to this shape.
//
// Q: [1, NQH, 32, DH]; K/V: [1, NKH, KV, DH] with NKH dividing NQH. Output: [1, NQH, 32, DH].
// `attn_mask` is accepted for call-site compatibility but treated as a no-op (non-causal full attention).
Tensor kv_sdpa(
    const Tensor& input_tensor_q,
    const Tensor& input_tensor_k,
    const Tensor& input_tensor_v,
    const std::optional<Tensor>& attn_mask = std::nullopt,
    std::optional<float> scale = std::nullopt,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttnn
