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
        One decode step of the gated delta rule with a fused gated RMSNorm, one core per value head. Plain mode runs
        B = 1 user; fused-conv mode (``conv_hist`` / ``conv_taps`` given) runs 1 <= B <= 32 users, user ``u`` in row
        ``u`` of every tile. For value head ``h`` (key head ``h // (Nv/Nk)``) and each user:
            qn = l2norm(q) * scale, kn = l2norm(k)
            S  = S * exp(g[h]);  delta = beta[h] * (v - kn @ S);  S += kn^T @ delta
            o  = qn @ S;  out = o / sqrt(mean(o^2) + norm_epsilon) * weight
        ``state`` is updated in place. In plain mode rows 1..31 of q/k/v are ignored (masked to zero).
        Args:
            qkv (ttnn.Tensor): BFLOAT16 TILE. Plain mode: ``[1, 1, 2*Nk*Dk + Nv*Dv]``, post conv+silu, laid out
                ``[q | k | v]``. Fused-conv mode: the projection rows ``[1, B, W]`` = ``[q | k | v | z | a | b]``.
            beta (ttnn.Tensor): FLOAT32 or BFLOAT16. Plain mode: ``[1, 1, Nv]`` update strengths (sigmoid already
                applied). Fused-conv mode: dt_bias, one row of Nv values (e.g. ``[1, 1, Nv]`` or ``[Nv]``).
            g (ttnn.Tensor): FLOAT32 or BFLOAT16. Plain mode: ``[1, 1, Nv]`` log decays. Fused-conv mode: -exp(A_log),
                one row of Nv values.
            state (ttnn.Tensor): ``[Bmax, Nv, Dk, Dv]`` FLOAT32 recurrent state, rows ``0..B-1`` updated in place
                (Bmax = 1 in plain mode, Bmax >= B in fused-conv mode).
            weight (ttnn.Tensor): BFLOAT16 gated-norm weight, one row of Dv values (e.g. ``[Dv]``).
            num_value_heads, num_key_heads, key_dim, value_dim (int).
        Keyword Args:
            scale (float, optional): query scale, defaults to ``key_dim ** -0.5``.
            l2_epsilon (float): l2-norm epsilon (default 1e-6). norm_epsilon (float): RMSNorm epsilon (default 1e-6).
            memory_config, compute_kernel_config, output_dtype (FLOAT32 or BFLOAT16, default BFLOAT16).
            conv_hist (ttnn.Tensor, optional): fused-conv mode: packed conv history ``[Bmax, Nv, 4, 32, 32]``
                BFLOAT16 (Bmax >= B), one tile per (user, value head, slot); channel chunk c of the head's
                ``[q | k | v]`` row for user u is in tile row ``2*c + (u & 1)``, slot 3 is the newest token. The op
                computes the 4-tap causal conv + SiLU, beta = sigmoid(b), decay = exp(-exp(A) * softplus(a + dt_bias)),
                gates the output with silu(z) and shifts each user's packed history in place (slot0 <- slot1, ...,
                slot3 <- new token).
            conv_taps (ttnn.Tensor, optional): packed taps ``[Nv, 4, 32, 32]`` BFLOAT16, one tile per (value head,
                tap), tap 0 = oldest; chunk c is duplicated in rows ``2*c`` and ``2*c + 1`` (one copy per user parity).
            qkvz_dim (int): column offset of the a|b block in the projection row (= 2*Nk*Dk + 2*Nv*Dv).
            fast_mode (bool): default False. Fused-conv mode only (needs conv_hist): the faster kernels (reads
                ordered by first use, beta/decay gates computed in fp32 on the reader, dual-output state add). Same
                output; the updated state keeps the full fp32 sum (the default path truncates it to TF32), so it
                differs in the last bits.
            fp32_state (bool): default False. fp32_state: keep the recurrent-state decay/add in exact fp32 on the SFPU (fast_mode only); avoids per-token state truncation.
            row_pos (ttnn.Tensor, optional): fused-conv mode: ``[B]`` INT32 ROW_MAJOR decode positions. Rows with a
                position < 0 keep their state and conv history (an idle decode row may hold a paused request).
        Returns:
            ttnn.Tensor: ``[1, B, Nv*Dv]`` normalized output; only logical rows ``0..B-1`` are valid.
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
        nb::arg("qkvz_dim") = 0,
        nb::arg("fast_mode") = false,
        nb::arg("fp32_state") = false,
        nb::arg("row_pos") = nb::none());
}

}  // namespace ttnn::operations::experimental::kda::gdn_decode_step::detail
