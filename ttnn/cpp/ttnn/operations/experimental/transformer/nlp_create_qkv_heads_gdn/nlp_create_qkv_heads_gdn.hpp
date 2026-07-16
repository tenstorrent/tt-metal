// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/nlp_create_qkv_heads_gdn_device_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// GDN fork of nlp_create_qkv_heads. Splits a fused token-major [B, 1, S, (Nq+Nk+Nv)*head_dim]
// input into head-major Q [B, Nq, S, head_dim], K [B, Nk, S, head_dim], V [B, Nv, S, head_dim].
// Q/K/V may each have an independent head count (GDN: Nq==Nk!=Nv); head_dim is shared and inferred.
std::tuple<Tensor, Tensor, Tensor> nlp_create_qkv_heads_gdn(
    const Tensor& input_tensor,
    uint32_t num_q_heads,
    uint32_t num_k_heads,
    uint32_t num_v_heads,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<std::vector<std::optional<Tensor>>>& optional_output_tensors = std::nullopt);

}  // namespace ttnn::experimental
