// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/deepseek/moe/generalized_moe_gate/generalized_moe_gate_nanobind.hpp"
#include <nanobind/nanobind.h>
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deepseek/moe/generalized_moe_gate/generalized_moe_gate.hpp"

namespace ttnn::operations::experimental::deepseek::moe::generalized_moe_gate::detail {

void bind_generalized_moe_gate(nb::module_& mod) {
    ttnn::bind_function<"generalized_moe_gate", "ttnn.experimental.deepseek.moe.">(
        mod,
        R"doc(
        Generalized (ungrouped) MoE gate routing on height-sharded tensors. Generalizes the DeepSeek-V3
        gate: scores the router logits (optionally via sigmoid), adds a selection bias, selects the
        top-``topk`` experts per token, normalizes their scores (softmax or linear renormalization), and
        applies ``scaling_factor``.

        Writes results into ``output_tensor`` and ``output_indices_tensor`` (same tensors are returned).

        Sharding / cores:
            All five tensors are HEIGHT_SHARDED in L1 with ROW_MAJOR orientation — one shard per core.
            There is no fixed grid: the op runs on exactly the cores covered by ``input_tensor``'s shard
            grid (the caller chooses it). Each core independently gates one token over all experts, so the
            number of cores equals the leading (token) shard dimension of the input — ``batch`` in the tests
            and model, one token per core; e.g. they allocate the grid via
            ``ttnn.num_cores_to_corerangeset(batch, compute_with_storage_grid_size(), row_wise=True)``.
            (A ``batch`` larger than the core count is the caller's concern — the model chunks it and reuses
            the cores across iterations, so this op only ever sees one iteration's worth of tokens.)
            ``bias_tensor`` and ``input_indices_tensor`` grids must contain the input grid; ``output_tensor``
            and ``output_indices_tensor`` must share one grid that also contains the input grid.

            Experts are packed ``num_blocks`` 32x32 tiles per shard, one 256-expert block per tile. Each
            block's 256 experts occupy ONLY the top-left 16x16 face (face0) of its 32x32 tile — the other
            three faces are unused padding (the SFPU top-k operates on a single face). ``num_blocks`` = 1 for
            experts <= 256, 2 for 256 < experts <= 512. There is no separate hidden/intermediate dim — a gate
            only sees [num_tokens, num_experts] and emits the top-``topk``.

        Shapes  (B = #tokens this call = #cores; E = #experts = 256 or 512; num_blocks = E/256, i.e. 1 or 2;
                 k = ``topk`` ∈ {4, 6, 8}). In the model the router logits arrive as ``[1, 1, B, E]`` and are
                 reshaped into the op's per-token block layout below:

            input_tensor / bias_tensor (BF16), input_indices_tensor (UInt16):
                logical    : rank >= 2, with the trailing two dims forming exactly one 256-expert block
                             (16x16). num_blocks is read from the SHARD shape, NOT the logical rank, so the
                             leading dims are just a token/block spelling and the op treats these uniformly:
                             the tests/model pass ``[B, 16, 16]`` for E <= 256 (num_blocks = 1) and
                             ``[B, num_blocks, 16, 16]`` for E <= 512 (num_blocks = 2) — same contract, the
                             rank only differs because the second form names the block dim explicitly.
                on device  : TILE layout, HEIGHT_SHARDED. Each 16x16 block sits in face0 of a 32x32 tile, so
                             the per-core shard is ``(num_blocks*32, 32)`` and the full (flattened) sharded
                             tensor is ``(B*num_blocks*32, 32)``.

            output_tensor (BF16), output_indices_tensor (UInt16):
                logical    : caller's choice — the op only requires a shape that pads to one 32x32 tile per
                             token (the (32, 32) shard) and constrains nothing else, so e.g. ``[B, 1, 16]``
                             and ``[B, 32, 32]`` are equivalent (neither is "bigger" — both are one tile).
                on device  : TILE layout, HEIGHT_SHARDED. Per-core shard ``(32, 32)`` (one tile/token), full
                             sharded tensor ``(B*32, 32)``. Only the first k entries of row 0 are valid (the
                             selected scores / expert ids); the model slices+views them back to ``[1, 1, B, k]``.

        Args:
            input_tensor: Router logits, BF16. Shard shape ``(num_blocks*32, 32)`` (num_blocks tiles stacked
                along the height); each 32x32 tile carries one 256-expert block in its top-left 16x16 face
                (face0) only — the rest of the tile is padding. One shard/token per core.
            bias_tensor: Score-correction bias added for selection only (output scores stay unbiased), BF16.
                Same shard spec / shape / orientation as ``input_tensor`` (transposed within each 16x16 block).
            input_indices_tensor: Routing indices (the global expert id per slot), UInt16. Same sharding as
                ``input_tensor`` — shard shape ``(num_blocks*32, 32)``, matching orientation, and a grid that
                contains the input grid; only the dtype and contents differ. One 32x32 tile per block holds
                that block's global ids (block b = arange(256) + b*256), transposed within each 16x16 block.
            output_tensor: Preallocated BF16 buffer for the normalized top-``topk`` scores. Shard shape
                ``(32, 32)`` — a single tile per token; only the first ``topk`` entries are valid.
            output_indices_tensor: Preallocated UInt16 buffer for the selected expert indices. Shard shape
                ``(32, 32)``, same grid as ``output_tensor``; only the first ``topk`` entries are valid.
            eps: Denominator stabilization for normalization (default: 1e-20).
            scaling_factor: Routed scaling factor applied after normalization (default: 2.5).
            enable_sigmoid: Apply sigmoid to the logits before the bias add when True (sigmoid routing);
                when False the raw logits are scored directly (default: False).
            topk: Number of experts selected per token. Supported values are 4, 6, or 8 ONLY — any other
                value is rejected by op validation (the finalize rank-mask handles exactly these; topk 1-3
                would leave ranks 0-3 unmasked and 5/7 are untested) (default: 8).
            output_softmax: Normalize the selected top-``topk`` scores with softmax when True; when False
                they are linearly renormalized (divided by their sum) (default: False).
            grouped: Run the DeepSeek grouped gate (8 groups × 32 -> top-2-sum per group -> top-4 groups ->
                top-8) instead of the ungrouped global top-k. Single 256-block (num_blocks == 1) only; forces
                top-8 + linear renormalization, so ``topk`` and ``output_softmax`` are ignored (default: False).

        Returns:
            Tuple ``(output_tensor, output_indices_tensor)`` — the normalized top-``topk`` scores and the
            selected expert indices.
        )doc",
        &ttnn::experimental::deepseek::moe::generalized_moe_gate,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("bias_tensor"),
        nb::arg("input_indices_tensor"),
        nb::arg("output_tensor"),
        nb::arg("output_indices_tensor"),
        nb::arg("eps") = 1e-20f,
        nb::arg("scaling_factor") = 2.5f,
        nb::arg("enable_sigmoid") = false,
        nb::arg("topk") = 8,
        nb::arg("output_softmax") = false,
        nb::arg("grouped") = false);
}

}  // namespace ttnn::operations::experimental::deepseek::moe::generalized_moe_gate::detail

namespace ttnn::operations::experimental::deepseek::moe::detail {

void bind_generalized_moe_gate(::nanobind::module_& mod) {
    generalized_moe_gate::detail::bind_generalized_moe_gate(mod);
}

}  // namespace ttnn::operations::experimental::deepseek::moe::detail
