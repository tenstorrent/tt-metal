// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_experts_prefill_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "fused_experts_prefill.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::fused_experts_prefill::detail {

void bind_fused_experts_prefill(nb::module_& mod) {
    ttnn::bind_function<"fused_experts_prefill", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        Routed-expert FFN for DeepSeek-V4-Flash prefill. Takes the decode ``fused_experts`` routing
        arguments and reads the DECODE weight layout in place (no second copy of the experts in DRAM).

        The routing is computed on device by every core: the k selected experts per token (from
        ``routing_indices``, or a top-k over ``ranking_scores``), their weights
        ``routed_scaling_factor * s / (sum(selected s) + routing_eps)`` (bf16-rounded, from the unbiased
        ``routing_scores``) and the per-expert token lists. Expert ``e`` runs on core group ``e % 15``
        (15 groups of 8 cores over the 12x10 grid); the group gathers the expert's token rows from
        ``x_tok`` (row-major DRAM), tilizes them, and for its tokens computes
        ``act = silu(min(gate, limit)) * clamp(up, -limit, limit)``, ``[gate, up] = x @ gate_up_e``,
        ``y = w * (act @ down_e)``. Rows are processed in M-blocks so every weight tile feeds several
        accumulators. The per-(slot, token) results are summed over the top_k slots.

        Args:
            x_tok (ttnn.Tensor): [1, 1, T, H] ROW_MAJOR BFLOAT16, DRAM interleaved. T is a multiple of
                32 and at most 512.
            routing_scores (ttnn.Tensor): [1, 1, T, E] TILE BFLOAT16, the unbiased scores.
            gate_up_weights (list[ttnn.Tensor]): E tensors [H, 2I], decode layout (DRAM ND-sharded,
                [H, 64] shards of one [gate_32 | up_32] pair).
            down_weights (list[ttnn.Tensor]): E tensors [I, H], decode layout (DRAM ND-sharded,
                [I, 64] shards).
            intermediate_size (int): local SwiGLU intermediate size I.
            swiglu_limit (float): SwiGLU clamp limit.
            top_k (int): experts per token, at most 16.
            routed_scaling_factor (float): scale applied to the renormalised weights.
            routing_eps (float): epsilon of the weight renormalisation.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): of the returned tensor.
            routing_indices (ttnn.Tensor, optional): [1, 1, T, top_k] TILE UINT16 / BFLOAT16 selected
                expert ids (valid and unique per token). Exactly one of this and ``ranking_scores``.
            ranking_scores (ttnn.Tensor, optional): [1, 1, T, E] TILE BFLOAT16 ranked on device.

        Returns:
            ttnn.Tensor: [1, 1, T, H] TILE BFLOAT16.
        )doc",
        &fused_experts_prefill,
        nb::arg("x_tok").noconvert(),
        nb::arg("routing_scores").noconvert(),
        nb::arg("gate_up_weights").noconvert(),
        nb::arg("down_weights").noconvert(),
        nb::arg("intermediate_size"),
        nb::arg("swiglu_limit"),
        nb::arg("top_k"),
        nb::arg("routed_scaling_factor"),
        nb::arg("routing_eps"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("routing_indices") = nb::none(),
        nb::arg("ranking_scores") = nb::none());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::fused_experts_prefill::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_fused_experts_prefill(::nanobind::module_& mod) {
    fused_experts_prefill::detail::bind_fused_experts_prefill(mod);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
