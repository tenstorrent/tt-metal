// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_hyperconnection_nanobind.hpp"

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deepseek/hyperconnection/fused_hyperconnection.hpp"

namespace ttnn::operations::experimental::deepseek::hyperconnection::detail {

void bind_fused_hyperconnection(nb::module_& mod) {
    ttnn::bind_function<"fused_hyperconnection", "ttnn.experimental.deepseek.">(
        mod,
        R"doc(
        Implements the ``pre`` / ``post`` / ``comb`` / ``collapsed`` portion of
        ``DeepSeekV4HyperConnection.forward`` given the packed linear projection ``fused_w``
        (shape ``[1, 1, T, (2+H)*H]``). The op splits ``fused_w`` into its ``pre_w`` /
        ``post_w`` / ``comb_w`` slices inside the ``fused_hyperconnection_pre_post`` device
        kernel; ``pre_w`` / ``post_w`` are consumed in-place and ``comb_w`` is returned
        already laid out as the ``[1, T, H, H]`` comb matrix. The RMSNorm + fn matmul that
        produces ``fused_w`` is NOT part of this op.

        The ``T = B*S`` tokens are independent and are spread across the core grid, so
        batched decode (``B > 1``) and multi-token prefill (``S > 1``) are supported.

            pre        = sigmoid(pre_w  * pre_scale  + pre_bias)  + eps
            post       = 2 * sigmoid(post_w * post_scale + post_bias)
            comb_logit = comb_w * comb_scale + comb_bias                (reshaped [1,T,H,H])
            comb       = sinkhorn(softmax(comb_logit, dim=-1) + eps, sinkhorn_iters)
            collapsed  = sum_h pre[..,h] * hidden_streams[..,h,:]

        If ``pre_mix`` is given, ``collapsed`` uses it in place of ``pre`` (DeepSeek V4.1, where
        each sublayer collapses with the ``pre`` computed by the sublayer before it); ``post`` and
        ``comb`` are unchanged, and ``pre`` itself is returned as a fourth output, in the layout
        ``pre_mix`` takes, for the next sublayer to collapse with.

        Args:
            hidden_streams: Residual-stream stack, [B, S, H, D].
            fused_w: Packed pre/post/comb projection output, [1, 1, T, (2+H)*H] (T == B*S).
            pre_bias: Bias row [1, 1, 1, H].
            post_bias: Bias row [1, 1, 1, H].
            comb_bias: Bias row [1, 1, 1, H*H].
            num_streams: Number of parallel streams H (config.hc_mult).
            sinkhorn_iters: Sinkhorn-Knopp iteration count (config.hc_sinkhorn_iters).
            pre_scale: Learned scale for the pre projection.
            post_scale: Learned scale for the post projection.
            comb_scale: Learned scale for the comb projection.
            eps: Stability epsilon added to pre / comb (config.hc_eps).
            memory_config: Optional output memory config.
            pre_mix: Optional collapse weights [B, S, 1, H], BFLOAT16 TILE (one tile per token).
                ``None`` collapses with the ``pre`` computed from ``fused_w``.

        Returns:
            List of [post [B,S,H,1], comb [B,S,H,H], collapsed [B,S,1,D]], plus
            pre [B,S,1,H] (BFLOAT16 TILE, zero-padded) when ``pre_mix`` is given.
        )doc",
        &ttnn::experimental::deepseek::hyperconnection::fused_hyperconnection,
        nb::arg("hidden_streams"),
        nb::kw_only(),
        nb::arg("fused_w"),
        nb::arg("pre_bias"),
        nb::arg("post_bias"),
        nb::arg("comb_bias"),
        nb::arg("num_streams"),
        nb::arg("sinkhorn_iters"),
        nb::arg("pre_scale"),
        nb::arg("post_scale"),
        nb::arg("comb_scale"),
        nb::arg("eps"),
        nb::arg("memory_config") = std::nullopt,
        nb::arg("pre_mix") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::deepseek::hyperconnection::detail

namespace ttnn::operations::experimental::deepseek::detail {

void bind_fused_hyperconnection(::nanobind::module_& mod) { hyperconnection::detail::bind_fused_hyperconnection(mod); }

}  // namespace ttnn::operations::experimental::deepseek::detail
