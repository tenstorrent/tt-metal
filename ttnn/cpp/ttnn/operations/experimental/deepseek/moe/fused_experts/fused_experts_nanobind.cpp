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
        into a single device operation. The expert selection is derived on-device (no host-side
        expert-id list). It comes from one of two mutually exclusive inputs:

        * ``routing_indices`` -- the router's own top-k output (each token's selected expert ids),
          passed through unmodified; and
        * ``ranking_scores`` -- the E-wide score row the router would have ranked (typically a
          bias-corrected copy of ``routing_scores``), with ``top_k`` given explicitly. The op then
          finds each token's top-k largest entries itself, inside the leader kernel that already
          reads the whole score row, so no separate ``ttnn.topk`` launch and no DRAM round-trip of
          its id output is needed.

        Either way the per-token weights are the *unbiased* ``routing_scores`` at the selected
        experts, gathered on device, renormalized to sum to 1 and scaled by ``routed_scaling_factor``
        -- so a caller never has to widen the selection into an E-wide weight row, a temporary built
        by a scatter + normalize + relayout chain purely for this op to scan it straight back down
        to k values.

        Returns a [1, B, H] BFLOAT16 tensor in the input's layout (TILE, or ROW_MAJOR when B == 1):
        act = silu(clamp(gate, max=limit)) * clamp(up, -limit, limit), [gate, up] = x @ gate_up_w[hit_ids[i]],
        and output[b] = sum_i w[b, hit_ids[i]] * (act[b] @ down_w[hit_ids[i]]); hit_ids are the
        routing-selected experts in ascending order and w the normalized weights above. The gate_up
        weights must be DRAM ND-sharded so each shard is one core's [H, 128] slice (gate/up columns
        interleaved at tile granularity), and the down weights DRAM ND-sharded so each shard is one
        core's [I, H/64] slice — both read in a single NoC read. The SwiGLU activation is gathered
        onto, and broadcast from, two hubs (the compute rectangle's opposite corners; core {0,0} is
        always one of them) so the down matmul can read it on every core.

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
            input_tensor: Activations, [1, 1, B, H] with B <= 32 token rows. TILE, ROW_MAJOR
                (B == 1 decode; loaded as 1x32 compute tiles, no tilize), or ROW_MAJOR
                HEIGHT_SHARDED L1 replicated over the compute grid -- the shard is the full
                ``[B, H]`` row, i.e. the tensor ``all_gather_for_matmul`` produces and
                ``matmul_decode`` consumes in place. A replicated row is already on every core, so
                the op neither reads it from DRAM nor broadcasts it, and its token count is the
                shard height (dim -2 carries ``B * num_cores``). The output uses the input's layout
                and, for a replicated input, defaults to DRAM interleaved.
            routing_scores: Unbiased per-expert scores, [1, 1, B, E] bfloat16 -- TILE, or ROW_MAJOR
                when B == 1 (decode; LinearDecode stick, no tilize). The tensor the selection
                indexes into. If the selection ranked by a bias-corrected copy, pass the uncorrected
                scores here -- those are the ones that become weights.
            gate_up_weights: List of [H, 2I] weight tensors, one per expert (all experts provided),
                with gate/up columns interleaved at tile (32-col) granularity.
            down_weights: List of [I, H] weight tensors, one per expert.
            num_experts: Number of routing-selected ("hit") experts to run: the size of the union of
                the tokens' selections. May be an upper bound, at the cost of one redundant weight
                fetch per unused slot.
            intermediate_size: SwiGLU intermediate size I.
            swiglu_limit: Clamp limit used by the SwiGLU activation.
            top_k: Experts selected per token row, at most 16 with ``routing_indices`` (the ids must
                fit one 16-wide tile face) and at most 32 with ``ranking_scores`` (the leader's
                running top-k lives on the RISC-V stack). 0 (the default) reads it from
                ``routing_indices``; it is required (and non-zero) with ``ranking_scores``.
            routed_scaling_factor: Scale applied after the per-token renormalize.
            routing_eps: Added to each token's score sum before dividing.
            experts_block_size: Experts to hold in L1 at once. 0 (the default) means all
                ``num_experts``, reproducing the single-block pipeline exactly. Because blocking
                double-buffers the activation block so consecutive blocks pipeline, the largest
                usable block is about half the largest usable single block.
            two_hub_gather: Split the per-block activation gather + broadcast across two hubs, the
                two opposite corners of the compute rectangle, each multicasting half the SwiGLU I
                dim on its own NoC. Halves the gather ingress and the broadcast egress per hub; set
                False for the original single-hub pipeline (one gather target, one multicast
                sender). Defaults to True.
            memory_config: Optional output memory config.
            routing_indices: Selected expert ids, [1, 1, B, top_k] TILE. Either uint16 (the index
                output of ``ttnn.topk``, passed through unmodified) or bfloat16 (a ``ttnn.embedding``
                gather from a frozen id table -- the only dtype that op gathers, and exact for
                E <= 256). Mutually exclusive with ``ranking_scores``.
            ranking_scores: Scores to rank on, [1, 1, B, E] bfloat16 -- TILE, or ROW_MAJOR when
                B == 1 (decode). Same shape and dtype rules as ``routing_scores``. When supplied the
                op selects each token's ``top_k`` largest entries on device and they become the hit
                ids; the weights still come from ``routing_scores`` at those ids, so "rank on the
                biased row, weight with the uncorrected one" is a single op. Mutually exclusive with
                ``routing_indices``; pass the very same tensor as ``routing_scores`` when ranking and
                weighting use one row (the op then reads it once). Ties are broken toward the lower
                expert id.
        )doc",
        &ttnn::experimental::deepseek::moe::fused_experts,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("routing_scores"),
        nb::arg("gate_up_weights"),
        nb::arg("down_weights"),
        nb::arg("num_experts"),
        nb::arg("intermediate_size"),
        nb::arg("swiglu_limit"),
        nb::arg("top_k") = 0,
        nb::arg("routed_scaling_factor") = 1.0F,
        nb::arg("routing_eps") = 0.0F,
        nb::arg("experts_block_size") = 0,
        nb::arg("two_hub_gather") = true,
        nb::arg("memory_config") = std::nullopt,
        nb::arg("routing_indices") = std::nullopt,
        nb::arg("ranking_scores") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::deepseek::moe::fused_experts::detail

namespace ttnn::operations::experimental::deepseek::moe::detail {

void bind_fused_experts(::nanobind::module_& mod) { fused_experts::detail::bind_fused_experts(mod); }

}  // namespace ttnn::operations::experimental::deepseek::moe::detail
