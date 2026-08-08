// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_experts_nanobind.hpp"

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deepseek/moe/fused_experts/fused_experts.hpp"

namespace ttnn::operations::experimental::deepseek::moe::fused_experts::detail {

void bind_fused_experts(nb::module_& mod) {
    ttnn::bind_function<"fused_experts", "ttnn.experimental.deepseek.moe.">(
        mod,
        R"doc(
        Experimental fused routed-expert FFN for DeepSeek V4-Flash, for a batch of B <= 32 tokens.

        Fuses the per-expert matmul -> SwiGLU -> matmul -> weighted-accumulate loop
        into a single device operation. Expert selection/scaling is read on-device from
        ``routing_weights``: for token b, the i-th weight pair is scaled by ``routing_weights[b, i]``,
        so experts with zero routing weight contribute nothing (no host-side expert-id list).

        Returns a [1, B, H] BFLOAT16 TILE tensor (the B token rows padded to a 32-row tile), where
        act = silu(clamp(gate, max=limit)) * clamp(up, -limit, limit), [gate, up] = x @ gate_up_w[hit_ids[i]],
        and output[b] = sum_i routing_weights[b, hit_ids[i]] * (act[b] @ down_w[hit_ids[i]]); hit_ids
        are the routing-selected experts in ascending order. The gate_up weights must be DRAM
        ND-sharded so each shard is one core's [H, 128] slice (gate/up columns interleaved at tile
        granularity), and the down weights DRAM ND-sharded so each shard is one core's [I, H/64]
        slice — both read in a single NoC read. The SwiGLU activation is gathered onto core {0,0} and
        broadcast to every core for the down matmul. ``input_tensor`` must be TILE layout and
        ``routing_weights`` ROW_MAJOR bfloat16.

        Batching: the B tokens are the rows of dim -2 and share one 32-row tile, so a [1, B, S, H]
        activation must be folded into [1, 1, B*S, H] first. The expert ids are the deduplicated
        *union* of the tokens' selections, so an expert several tokens picked is fetched from DRAM
        once and its matmuls run once for the whole batch; weight traffic — the op's bottleneck —
        scales with the number of distinct experts, not with the token count.

        Expert blocking: ``num_experts`` is not bounded by L1. ``experts_block_size`` sets how many
        experts' activations are gathered and held at once, and the experts run in blocks of that
        size, which is what makes a batch of tokens selecting *disjoint* experts feasible (32 tokens
        at top_k 6 select up to 192 distinct experts, far more activation than fits in L1 at once).
        Blocking changes no arithmetic and no DRAM traffic — each expert is still fetched exactly
        once — at the cost of one gather/broadcast synchronization per block.

        Args:
            input_tensor: Activations, [1, 1, B, H] with B <= 32 token rows.
            routing_weights: Per-token routing weights, [1, 1, B, E] (ROW_MAJOR bfloat16),
                with E == len(gate_up_weights).
            gate_up_weights: List of [H, 2I] weight tensors, one per expert (all experts provided),
                with gate/up columns interleaved at tile (32-col) granularity.
            down_weights: List of [I, H] weight tensors, one per expert.
            num_experts: Number of routing-selected ("hit") experts to run: the number of columns
                nonzero in at least one routing row (the union over tokens). May be an upper bound,
                at the cost of one redundant weight fetch per unused slot.
            intermediate_size: SwiGLU intermediate size I.
            swiglu_limit: Clamp limit used by the SwiGLU activation.
            experts_block_size: Experts to hold in L1 at once. 0 (the default) means all
                ``num_experts``, reproducing the single-block pipeline exactly. Because blocking
                double-buffers the activation block so consecutive blocks pipeline, the largest
                usable block is about half the largest usable single block.
            memory_config: Optional output memory config.
        )doc",
        &ttnn::experimental::deepseek::moe::fused_experts,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("routing_weights"),
        nb::arg("gate_up_weights"),
        nb::arg("down_weights"),
        nb::arg("num_experts"),
        nb::arg("intermediate_size"),
        nb::arg("swiglu_limit"),
        nb::arg("experts_block_size") = 0,
        nb::arg("memory_config") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::deepseek::moe::fused_experts::detail

namespace ttnn::operations::experimental::deepseek::moe::detail {

void bind_fused_experts(::nanobind::module_& mod) { fused_experts::detail::bind_fused_experts(mod); }

}  // namespace ttnn::operations::experimental::deepseek::moe::detail
