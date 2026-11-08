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
            fast_mode (bool): default False. Fused-conv mode only (needs conv_hist): the faster kernels (reads
                ordered by first use, beta/decay gates computed in fp32 on the reader, dual-output state add). Same
                output; the updated state keeps the full fp32 sum (the default path truncates it to TF32), so it
                differs in the last bits.
            fp32_state (bool): default False. fp32_state: keep the recurrent-state decay/add in exact fp32 on the SFPU (fast_mode only); avoids per-token state truncation.
            state_page_idx (ttnn.Tensor, optional): fused-conv mode only. ``[2*Nv, >= B]`` ROW_MAJOR INT32 or UINT32
                device tensor that makes ``state`` and ``conv_hist`` page pools (for example views of vLLM-owned cache
                blocks). Column ``b``, row ``h`` is the first FLOAT32 page of value head ``h`` of batch row ``b``
                (pages ``[idx[h][b], idx[h][b] + KV)``, KV = Dk*Dv/1024, tile order as in ``[Dk, Dv]``); row ``Nv + h``
                is the first BFLOAT16 page of its packed history (pages ``idx[Nv + h][b] + slot``). Each head may sit
                anywhere, at any page alignment. ``state`` and ``conv_hist`` may then have any tiled shape. In this mode
                the packed history holds channel chunk c in both tile rows ``2c`` and ``2c + 1`` (as ``conv_taps``
                does; without the index only row ``2c + (b & 1)``), and the op keeps both when it shifts in the new
                token, so a request may sit on batch rows of either parity from step to step. The rows' page
                ranges must not overlap. The values are read on device on every call, so a captured trace follows
                new pages when only the contents of ``state_page_idx`` change. Default None: row ``b`` uses rows ``b``
                of ``state`` ``[Bmax, Nv, Dk, Dv]`` and ``conv_hist`` ``[Bmax, Nv, 4, 32, 32]``.
            state_pool2 (ttnn.Tensor, optional): with ``state_page_idx`` only. A second FLOAT32 tiled state pool: a
                state index value ``p >= P`` (P = page count of ``state``) addresses page ``p - P`` of ``state_pool2``,
                so the heads of one row can be split over two buffers (for example the K and V cache of a layer). A
                head's KV pages must lie in one pool.
            conv_hist_pool2 (ttnn.Tensor, optional): the same for the history: a BFLOAT16 tiled pool addressed by
                history index values past the page count of ``conv_hist``.
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
        nb::arg("qkvz_dim") = 0,
        nb::arg("fast_mode") = false,
        nb::arg("fp32_state") = false,
        nb::arg("state_page_idx") = nb::none(),
        nb::arg("state_pool2") = nb::none(),
        nb::arg("conv_hist_pool2") = nb::none());
}

}  // namespace ttnn::operations::experimental::kda::gdn_decode_step::detail
