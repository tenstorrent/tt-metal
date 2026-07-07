// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::transformer {

// Fused preprocessing for gated_delta_attn_seq. Inputs are already padded/chunkable,
// scaled, head-major tensors:
//   q/k/v     [BH, L, 128]
//   beta/g    [BH, L, 1]
// cached masks are [1, 128, 128] except eye_32 [1, 32, 32].
//
// Returns, in order, the tensors consumed by gated_delta_attn_seq:
//   L_unit, v_beta_sc, k_bd_sc, intra_attn, q_decay, k_decay_t, dl_exp, L_inv.
std::vector<ttnn::Tensor> gated_delta_attn_preprocess(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& beta,
    const ttnn::Tensor& g,
    const ttnn::Tensor& triu_ones,
    const ttnn::Tensor& tril_mask,
    const ttnn::Tensor& eye,
    const ttnn::Tensor& lower_causal,
    const ttnn::Tensor& eye_32,
    uint32_t chunk_size = 128,
    float diag_alpha = 0.25f,
    bool bf16_value_path = false,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::transformer
