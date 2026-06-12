// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/deepseek/moe/generalized_moe_gate/generalized_moe_gate_nanobind.hpp"

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

        Args:
            input_tensor: Router logits per shard ([*, 16, 16] logical = 256 experts, tile 32x32).
            bias_tensor: Score-correction bias added for selection only (output scores stay unbiased),
                same shard spec as logits.
            input_indices_tensor: Transposed routing indices shard (the global expert id per slot).
            output_tensor: Preallocated BF16 buffer for the normalized top-``topk`` scores (shard 32x32).
            output_indices_tensor: Preallocated UInt16 buffer for the selected expert indices (shard 32x32).
            eps: Denominator stabilization for normalization (default: 1e-20).
            scaling_factor: Routed scaling factor applied after normalization (default: 2.5).
            enable_sigmoid: Apply sigmoid to the logits before the bias add when True (sigmoid routing);
                when False the raw logits are scored directly (default: False).
            topk: Number of experts selected per token; supported values are 4, 6, 8 (default: 8).
            output_softmax: Normalize the selected top-``topk`` scores with softmax when True; when False
                they are linearly renormalized (divided by their sum) (default: False).

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
        nb::arg("output_softmax") = false);
}

}  // namespace ttnn::operations::experimental::deepseek::moe::generalized_moe_gate::detail

namespace ttnn::operations::experimental::deepseek::moe::detail {

void bind_generalized_moe_gate(::nanobind::module_& mod) {
    generalized_moe_gate::detail::bind_generalized_moe_gate(mod);
}

}  // namespace ttnn::operations::experimental::deepseek::moe::detail
