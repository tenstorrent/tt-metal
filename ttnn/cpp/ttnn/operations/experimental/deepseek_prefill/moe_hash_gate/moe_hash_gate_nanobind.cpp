// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_hash_gate_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/array.h>
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/moe_hash_gate/moe_hash_gate.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_hash_gate::detail {

void bind_moe_hash_gate(nb::module_& mod) {
    ttnn::bind_function<"moe_hash_gate", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            DeepSeek-V4 hash-routing MoE gate. Expert selection is a frozen tid2eid[input_ids] lookup
            (fused into the reader kernel) rather than a learned top-k. To each gate logit it applies
            score_func (sqrt(softplus(x)) for DeepSeek-V4, or sigmoid), gathers the activated scores at
            the looked-up expert indices, normalizes them across the selected experts, and scales by
            route_scale. Shares the activation / gather / normalize / scale kernels with moe_grouped_topk.

            Args:
                scores (ttnn.Tensor): Gate logits (dtype FLOAT32, layout TILE). Shape [..., n_routed_experts].
                input_ids (ttnn.Tensor): Per-token vocabulary ids (dtype UINT32, layout ROW_MAJOR), one per
                    score row, shaped [num_score_tiles_along_tokens, 32].
                tid2eid (ttnn.Tensor): Frozen token-id -> expert-id table (dtype UINT16, layout ROW_MAJOR),
                    one row per token id; the first n_activated_experts columns are used (row may be padded).
                n_activated_experts (int): Number of experts selected per token (must be <= 32).
                route_scale (float): Scale applied to the normalized scores. Defaults to 1.0.
                epsilon (float): Epsilon for numerical stability when normalizing. Defaults to 1e-20.
                score_func (str): Router affinity activation. "sqrtsoftplus" (DeepSeek-V4, default) or "sigmoid".
                memory_config (ttnn.MemoryConfig, optional): Output memory configuration. Defaults to None.
                padding_config (ttnn.Tensor, optional): ROW_MAJOR UINT32 tensor with per-device
                    [num_real_tokens, pad_side]. pad_side 0 = right padding, 1 = left padding. Defaults to None.

            Returns:
                Tuple[ttnn.Tensor, ttnn.Tensor]: scaled expert scores (dtype BFLOAT16) and selected expert
                indices (dtype UINT16), each shaped [..., n_activated_experts].
        )doc",
        &ttnn::operations::experimental::deepseek_prefill::moe_hash_gate::moe_hash_gate,
        nb::arg("scores").noconvert(),
        nb::arg("input_ids").noconvert(),
        nb::arg("tid2eid").noconvert(),
        nb::kw_only(),
        nb::arg("n_activated_experts"),
        nb::arg("route_scale") = 1.0f,
        nb::arg("epsilon") = 1e-20f,
        nb::arg("score_func") = "sqrtsoftplus",
        nb::arg("memory_config") = nb::none(),
        nb::arg("padding_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_hash_gate::detail
