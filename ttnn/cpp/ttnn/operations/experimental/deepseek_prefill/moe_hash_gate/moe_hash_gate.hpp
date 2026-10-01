// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_hash_gate {

// DeepSeek-V4 hash-routing gate. Expert selection is a static tid2eid[input_ids] lookup (done in the
// reader kernel), not a top-k; the per-expert weights are still score_func(scores) gathered at those
// indices, normalized, and scaled. Shares the activation/gather/normalize/scale kernel blocks with
// moe_grouped_topk.
//
//   scores  (FLOAT32, TILE): gate logits, shape [..., n_routed_experts].
//   input_ids (UINT32): per-token vocabulary ids, one per score row.
//   tid2eid (UINT16, ROW_MAJOR): frozen token-id -> expert-id table, row-per-token-id, first
//           n_activated_experts columns valid (row padded for NoC alignment).
//   score_func: "sqrtsoftplus" (DeepSeek-V4, default) or "sigmoid".
std::array<Tensor, 2> moe_hash_gate(
    const Tensor& scores,
    const Tensor& input_ids,
    const Tensor& tid2eid,
    uint32_t n_activated_experts,
    float route_scale = 1.0f,
    float epsilon = 1e-20f,
    const std::string& score_func = "sqrtsoftplus",
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& padding_config = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_hash_gate
