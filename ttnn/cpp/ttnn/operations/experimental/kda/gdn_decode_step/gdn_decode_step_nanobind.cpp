// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "gdn_decode_step_nanobind.hpp"

#include "gdn_decode_step.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::kda::gdn_decode_step::detail {

void bind_gdn_decode_step(nb::module_& mod) {
    ttnn::bind_function<"gdn_decode_step", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        One decode step (B = 1) of the gated delta rule with a fused gated RMSNorm, one core per value head.
        For value head ``h`` (key head ``h // (Nv/Nk)``), with the token in row 0 of every tile:
            qn = l2norm(q) * scale, kn = l2norm(k)
            S  = S * exp(g[h]);  delta = beta[h] * (v - kn @ S);  S += kn^T @ delta
            o  = qn @ S;  out = o / sqrt(mean(o^2) + norm_epsilon) * weight
        ``state`` is updated in place. Rows 1..31 of q/k/v are ignored (masked to zero).
        Args:
            qkv (ttnn.Tensor): ``[1, 1, 2*Nk*Dk + Nv*Dv]`` BFLOAT16 TILE, post conv+silu, laid out ``[q | k | v]``.
            beta (ttnn.Tensor): ``[1, 1, Nv]`` FLOAT32 or BFLOAT16 update strengths (sigmoid already applied).
            g (ttnn.Tensor): ``[1, 1, Nv]`` FLOAT32 or BFLOAT16 log decays.
            state (ttnn.Tensor): ``[1, Nv, Dk, Dv]`` FLOAT32 recurrent state, updated in place.
            weight (ttnn.Tensor): ``[Dv]`` BFLOAT16 gated-norm weight.
            num_value_heads, num_key_heads, key_dim, value_dim (int).
        Keyword Args:
            scale (float, optional): query scale, defaults to ``key_dim ** -0.5``.
            l2_epsilon (float): l2-norm epsilon (default 1e-6). norm_epsilon (float): RMSNorm epsilon (default 1e-6).
            memory_config, compute_kernel_config, output_dtype (FLOAT32 or BFLOAT16, default BFLOAT16).
            conv_hist (ttnn.Tensor, optional): fused-conv mode: packed conv history ``[Nv, 4, 32, 32]`` BFLOAT16, one
                tile per (value head, slot); row c of a tile is channel chunk c of the head's ``[q | k | v]`` row, slot 3
                is the newest token. ``qkv`` is then the full projection row ``[1, 1, W]`` = ``[q | k | v | z | a | b]``,
                ``beta`` is dt_bias and ``g`` is -exp(A_log) (both volume Nv). The op computes the 4-tap causal conv +
                SiLU, beta = sigmoid(b), decay = exp(-exp(A) * softplus(a + dt_bias)), gates the output with silu(z) and
                shifts the packed history in place (slot0 <- slot1, ..., slot3 <- new token).
            conv_taps (ttnn.Tensor, optional): packed taps ``[Nv, 4, 32, 32]`` BFLOAT16 in the same layout (tap 0 = oldest).
            qkvz_dim (int): column offset of the a|b block in the projection row (= 2*Nk*Dk + 2*Nv*Dv).
        Returns:
            ttnn.Tensor: ``[1, 1, Nv*Dv]`` normalized output (row 0 valid, padding rows zero).
        )doc",
        &ttnn::experimental::kda::gdn_decode_step,
        nb::arg("qkv").noconvert(),
        nb::arg("beta").noconvert(),
        nb::arg("g").noconvert(),
        nb::arg("state").noconvert(),
        nb::arg("weight").noconvert(),
        nb::arg("num_value_heads"),
        nb::arg("num_key_heads"),
        nb::arg("key_dim"),
        nb::arg("value_dim"),
        nb::kw_only(),
        nb::arg("scale") = nb::none(),
        nb::arg("l2_epsilon") = 1e-6f,
        nb::arg("norm_epsilon") = 1e-6f,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("output_dtype") = ttnn::DataType::BFLOAT16,
        nb::arg("conv_hist") = nb::none(),
        nb::arg("conv_taps") = nb::none(),
        nb::arg("qkvz_dim") = 0);
}

}  // namespace ttnn::operations::experimental::kda::gdn_decode_step::detail
