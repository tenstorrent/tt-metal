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
        One gated delta rule decode step for ``B <= 32`` batch rows with the causal convolution, the gates and a
        gated RMSNorm fused in. One core handles one value head for a group of batch rows. For batch row ``b`` and
        value head ``h`` (key head ``h // (Nv/Nk)``):
            x  = silu(sum_j taps[j] * hist[j]), hist = the 3 previous [q | k | v] rows and the new one
            qn = l2norm(q) * scale, kn = l2norm(k), beta = sigmoid(b), decay = exp(g[h] * softplus(a + dt_bias[h]))
            S  = S * decay;  delta = beta * (v - kn @ S);  S += kn^T @ delta
            o  = qn @ S;  out = o / sqrt(mean(o^2) + norm_epsilon) * weight * silu(z)
        ``state`` and ``conv_hist`` are updated in place.
        Args:
            qkv (ttnn.Tensor): ``[1, B, W]`` BFLOAT16 TILE projection rows ``[q | k | v | z | a | b]``.
            beta (ttnn.Tensor): dt_bias, FLOAT32 or BFLOAT16 TILE, volume Nv.
            g (ttnn.Tensor): -exp(A_log), FLOAT32 or BFLOAT16 TILE, volume Nv.
            state (ttnn.Tensor): ``[Bmax >= B, Nv, Dk, Dv]`` FLOAT32 recurrent state, updated in place (a page pool
                with ``state_page_idx``).
            weight (ttnn.Tensor): ``[Dv]`` BFLOAT16 gated-norm weight.
            num_value_heads, num_key_heads, key_dim, value_dim (int).
        Keyword Args:
            scale (float, optional): query scale, defaults to ``key_dim ** -0.5``.
            l2_epsilon (float): l2-norm epsilon (default 1e-6). norm_epsilon (float): RMSNorm epsilon (default 1e-6).
            memory_config, compute_kernel_config, output_dtype (FLOAT32 or BFLOAT16, default BFLOAT16).
            conv_hist (ttnn.Tensor): required. Packed conv history ``[Bmax >= B, Nv, 4, 32, 32]`` BFLOAT16, one tile
                per (batch row, value head, slot); slot 3 is the newest token. Batch row ``b`` keeps channel chunk
                ``c`` of the head's ``[q | k | v]`` row in tile row ``2c + (b & 1)``. The op reads slots 1..3 and
                shifts the history in place (slot 0 <- slot 1, ..., slot 3 <- new token); it never reads slot 0.
            conv_taps (ttnn.Tensor): required. Packed taps ``[Nv, 4, 32, 32]`` BFLOAT16, tap 0 = oldest, channel chunk
                ``c`` in tile rows ``2c`` and ``2c + 1``.
            qkvz_dim (int): column offset of the a|b block in the projection row (= 2*Nk*Dk + 2*Nv*Dv).
            fast_mode (bool): default False. The faster kernels (reads ordered by first use, beta/decay gates
                computed in fp32 on the reader, dual-output state add). Same output; the updated state keeps the full
                fp32 sum (the default path truncates it to TF32), so it differs in the last bits.
            fp32_state (bool): default False, needs fast_mode. Keep the recurrent-state decay and add in exact fp32
                on the SFPU, so the state is not truncated every token.
            state_page_idx (ttnn.Tensor, optional): ``[2*Nv, >= B]`` ROW_MAJOR INT32 or UINT32 device tensor that makes ``state`` and ``conv_hist`` page pools (for example views of vLLM-owned cache
                blocks). Column ``b``, row ``h`` is the first FLOAT32 page of value head ``h`` of batch row ``b``
                (pages ``[idx[h][b], idx[h][b] + KV)``, KV = Dk*Dv/1024, tile order as in ``[Dk, Dv]``); row ``Nv + h``
                is the first BFLOAT16 page of its packed history (pages ``idx[Nv + h][b] + slot``). Each head may sit
                anywhere, at any page alignment. ``state`` and ``conv_hist`` may then have any tiled shape. In this mode
                the packed history holds channel chunk c in both tile rows ``2c`` and ``2c + 1`` (as ``conv_taps``
                does; without the index only row ``2c + (b & 1)``), and the op keeps both when it shifts in the new
                token, so a request may sit on batch rows of either parity from step to step. The rows' page
                ranges must not overlap, and the op does not range-check the values. The values are read on device on every call, so a captured trace follows
                new pages when only the contents of ``state_page_idx`` change. Default None: row ``b`` uses rows ``b``
                of ``state`` ``[Bmax, Nv, Dk, Dv]`` and ``conv_hist`` ``[Bmax, Nv, 4, 32, 32]``.
            state_pool2 (ttnn.Tensor, optional): with ``state_page_idx`` only. A second FLOAT32 tiled state pool: a
                state index value ``p >= P`` (P = page count of ``state``) addresses page ``p - P`` of ``state_pool2``,
                so the heads of one row can be split over two buffers (for example the K and V cache of a layer). A
                head's KV pages must lie in one pool.
            conv_hist_pool2 (ttnn.Tensor, optional): the same for the history: a BFLOAT16 tiled pool addressed by
                history index values past the page count of ``conv_hist``.
        Returns:
            ttnn.Tensor: ``[1, B, Nv*Dv]`` gated, normalized output, one row per batch row.
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
