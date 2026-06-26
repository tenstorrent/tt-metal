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
        Experimental fused Manifold-Constrained Hyper-Connection (mHC) post-projection stage.

        Implements the ``pre`` / ``post`` / ``comb`` / ``collapsed`` portion of
        ``DeepSeekV4HyperConnection.forward`` (lines 87-108) given the three already-computed
        linear projections ``pre_w`` / ``post_w`` / ``comb_w``. The RMSNorm + fn matmuls that
        produce them are NOT part of this op.

            pre        = sigmoid(pre_w  * pre_scale  + pre_bias)  + eps
            post       = 2 * sigmoid(post_w * post_scale + post_bias)
            comb_logit = comb_w * comb_scale + comb_bias                (reshaped [1,T,H,H])
            comb       = sinkhorn(softmax(comb_logit, dim=-1) + eps, sinkhorn_iters)
            collapsed  = sum_h pre[..,h] * hidden_streams[..,h,:]

        Args:
            hidden_streams: Residual-stream stack, [B, S, H, D].
            pre_w: Pre projection output, [1, 1, T, H] (T == B*S).
            post_w: Post projection output, [1, 1, T, H].
            comb_w: Comb projection output, [1, 1, T, H*H].
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

        Returns:
            Tuple of (post [B,S,H,1], comb [B,S,H,H], collapsed [B,S,1,D]).
        )doc",
        &ttnn::experimental::deepseek::hyperconnection::fused_hyperconnection,
        nb::arg("hidden_streams"),
        nb::kw_only(),
        nb::arg("pre_w"),
        nb::arg("post_w"),
        nb::arg("comb_w"),
        nb::arg("pre_bias"),
        nb::arg("post_bias"),
        nb::arg("comb_bias"),
        nb::arg("num_streams"),
        nb::arg("sinkhorn_iters"),
        nb::arg("pre_scale"),
        nb::arg("post_scale"),
        nb::arg("comb_scale"),
        nb::arg("eps"),
        nb::arg("memory_config") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::deepseek::hyperconnection::detail

namespace ttnn::operations::experimental::deepseek::detail {

void bind_fused_hyperconnection(::nanobind::module_& mod) { hyperconnection::detail::bind_fused_hyperconnection(mod); }

}  // namespace ttnn::operations::experimental::deepseek::detail
