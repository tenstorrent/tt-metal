// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_expert_ffn_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "routed_expert_ffn.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::detail {

void bind_routed_expert_ffn(nb::module_& mod) {
    ttnn::bind_function<"routed_expert_ffn", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        Single expert FFN computation for DeepSeek MoE prefill.

        Computes the gated FFN for one routed expert:
            gate_out = x @ gate_proj
            up_out   = x @ up_proj
            activated = silu(gate_out) * up_out
            output   = activated @ down_proj

        Args:
            x (ttnn.Tensor): Input tensor.
            gate_proj (ttnn.Tensor): Gate projection weight (emb_dim, hidden_dim).
            up_proj (ttnn.Tensor): Up projection weight (emb_dim, hidden_dim).
            down_proj (ttnn.Tensor): Down projection weight (hidden_dim, emb_dim).

        Keyword Args:
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Compute kernel configuration. Defaults to None.
            output (ttnn.Tensor, optional): Pre-allocated output tensor for in-place write of the final matmul result. Defaults to None.
            global_expert_idx_table (ttnn.Tensor, optional): DRAM uint32 TILE_LAYOUT tensor of shape (1, 1, experts_per_chip) mapping local expert slots to global expert ids. When paired with ``expert_token_counts``, enables the Blackhole routed matmul path whose per-chunk guard skips iff ``expert_token_counts[global_expert_idx_table[local_expert_idx]] <= curr_expert_iter * expert_iter_length``. Falls back to ttnn::matmul when either tensor is None.
            expert_token_counts (ttnn.Tensor, optional): DRAM uint32 TILE_LAYOUT tensor of shape (1, 1, num_global_experts). See ``global_expert_idx_table`` for semantics.
            local_expert_idx (int, optional): Index into ``global_expert_idx_table``. Defaults to 0.
            curr_expert_iter (int, optional): Index of the current chunk iteration (Blackhole only). Defaults to 0.
            expert_iter_length (int, optional): Tokens per chunk iteration (Blackhole only). Defaults to 0.

        Returns:
            ttnn.Tensor: Output tensor with the same shape as ``x``.

        Example:
            >>> output = ttnn.experimental.deepseek_prefill.routed_expert_ffn(
                    x, gate_proj, up_proj, down_proj,
                    compute_kernel_config=compute_kernel_config,
                    global_expert_idx_table=table,
                    expert_token_counts=counts,
                    local_expert_idx=local_idx,
                    curr_expert_iter=chunk_idx,
                    expert_iter_length=2048)
        )doc",
        &routed_expert_ffn,
        nb::arg("x").noconvert(),
        nb::arg("gate_proj").noconvert(),
        nb::arg("up_proj").noconvert(),
        nb::arg("down_proj").noconvert(),
        nb::kw_only(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("output") = nb::none(),
        nb::arg("global_expert_idx_table") = nb::none(),
        nb::arg("expert_token_counts") = nb::none(),
        nb::arg("local_expert_idx") = 0u,
        nb::arg("curr_expert_iter") = 0u,
        nb::arg("expert_iter_length") = 0u);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_routed_expert_ffn(::nanobind::module_& mod) { routed_expert_ffn::detail::bind_routed_expert_ffn(mod); }

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
