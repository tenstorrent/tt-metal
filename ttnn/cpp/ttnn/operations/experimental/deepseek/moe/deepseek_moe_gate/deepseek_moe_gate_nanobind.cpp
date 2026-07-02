// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/deepseek/moe/deepseek_moe_gate/deepseek_moe_gate_nanobind.hpp"

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deepseek/moe/deepseek_moe_gate/deepseek_moe_gate.hpp"

namespace ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate::detail {

void bind_deepseek_moe_gate(nb::module_& mod) {
    ttnn::bind_function<"deepseek_moe_gate", "ttnn.experimental.deepseek.moe.">(
        mod,
        R"doc(
        DeepSeek V3 MoE gate routing (top-8 experts with normalized scores) on height-sharded tensors.

        Writes results into ``output_tensor`` and ``output_indices_tensor`` (same tensors are returned).

        Args:
            input_tensor: Router logits per shard ([*, 16, 16] logical, tile 32x32).
            bias_tensor: Score correction bias, same shard spec as logits.
            input_indices_tensor: Transposed routing indices shard.
            output_tensor: Preallocated BF16 buffer for normalized top-8 scores (shard 32x32).
            output_indices_tensor: Preallocated UInt16 indices buffer (shard 32x32).
            eps: Denominator stabilization for normalization (default: 1e-20).
            scaling_factor: Routed scaling factor applied after normalization (default: 2.5).
            enable_sigmoid: Apply sigmoid to logits before bias add when True (default: False).

        Returns:
            Tuple ``(output_tensor, output_indices_tensor)``.
        )doc",
        &ttnn::experimental::deepseek::moe::deepseek_moe_gate,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("bias_tensor"),
        nb::arg("input_indices_tensor"),
        nb::arg("output_tensor"),
        nb::arg("output_indices_tensor"),
        nb::arg("eps") = 1e-20f,
        nb::arg("scaling_factor") = 2.5f,
        nb::arg("enable_sigmoid") = false);
}

}  // namespace ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate::detail

namespace ttnn::operations::experimental::deepseek::moe::detail {

void bind_deepseek_moe_gate(::nanobind::module_& mod) { deepseek_moe_gate::detail::bind_deepseek_moe_gate(mod); }

}  // namespace ttnn::operations::experimental::deepseek::moe::detail
