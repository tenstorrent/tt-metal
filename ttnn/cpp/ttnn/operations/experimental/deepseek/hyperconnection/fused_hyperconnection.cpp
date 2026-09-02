// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_hyperconnection.hpp"

#include "device/fused_pre_post_device_operation.hpp"
#include "device/fused_single_user_device_operation.hpp"
#include "device/sinkhorn_device_operation.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/creation/creation.hpp"
namespace ttnn::experimental::deepseek::hyperconnection {

std::tuple<Tensor, Tensor, Tensor> fused_hyperconnection(
    const Tensor& hidden_streams,
    const Tensor& fused_w,
    const Tensor& pre_bias,
    const Tensor& post_bias,
    const Tensor& comb_bias,
    uint32_t num_streams,
    uint32_t sinkhorn_iters,
    float pre_scale,
    float post_scale,
    float comb_scale,
    float eps,
    const std::optional<MemoryConfig>& memory_config) {
    const auto& shape = hidden_streams.logical_shape();
    const uint32_t b = static_cast<uint32_t>(shape[0]);
    const uint32_t s = static_cast<uint32_t>(shape[1]);
    const uint32_t hc = num_streams;
    const uint32_t d = static_cast<uint32_t>(shape[-1]);
    const uint32_t num_tokens = b * s;

    if (num_tokens == 1) {
        // Single-user decode has a dedicated multi-core program: cores 0..7
        // compute the width-sharded collapse, core 8 computes post, and core 9
        // computes comb plus Sinkhorn.
        Tensor comb_bias_mat = ttnn::reshape(comb_bias, ttnn::Shape({1, 1, hc, hc}));
        auto [post, comb, collapsed] = ttnn::prim::fused_hyperconnection_single_user(
            fused_w,
            pre_bias,
            post_bias,
            comb_bias_mat,
            hidden_streams,
            hc,
            sinkhorn_iters,
            pre_scale,
            post_scale,
            comb_scale,
            eps,
            memory_config);
        return {post, comb, collapsed};
    }

    // Fused stage over all T = B*S tokens:
    //   post      = 2 * sigmoid(post_w * post_scale + post_bias)            [1,T,H,1]
    //   collapsed = (sigmoid(pre_w * pre_scale + pre_bias) + eps) @ hidden  [1,T,1,D]
    //   comb_w_mat = comb_w slice of fused_w, laid out as the HxH grid      [1,T,H,H]
    // The pre/post/comb slices are split out of `fused_w` inside the device op; pre_w / post_w
    // are consumed in-place, comb_w is returned already in the grid layout the Sinkhorn stage
    // expects (no host-side reshape).
    auto [post, collapsed, comb_w_mat] = ttnn::prim::fused_hyperconnection_pre_post(
        fused_w, pre_bias, post_bias, hidden_streams, hc, pre_scale, post_scale, eps, memory_config);

    // comb: softmax(comb_w * comb_scale + comb_bias, dim=-1) + eps, then Sinkhorn (alternate
    // row/col normalisation) onto the doubly-stochastic manifold, fused into a single device op.
    // comb_bias is [1,1,1,H*H]; reshape to the [1,1,H,H] comb matrix the op broadcasts over tokens.
    Tensor comb_bias_mat = ttnn::reshape(comb_bias, ttnn::Shape({1, 1, hc, hc}));
    Tensor comb = ttnn::prim::fused_hyperconnection_sinkhorn(
        comb_w_mat, comb_bias_mat, hc, sinkhorn_iters, comb_scale, eps, memory_config);

    // The device ops keep tokens on dim 1 with the trailing two dims already in their final
    // layout, so splitting T back into (B, S) is a metadata-only reshape.
    post = ttnn::reshape(post, ttnn::Shape({b, s, hc, 1}));
    comb = ttnn::reshape(comb, ttnn::Shape({b, s, hc, hc}));
    collapsed = ttnn::reshape(collapsed, ttnn::Shape({b, s, 1, d}));

    if (memory_config.has_value()) {
        post = ttnn::to_memory_config(post, *memory_config);
        comb = ttnn::to_memory_config(comb, *memory_config);
        collapsed = ttnn::to_memory_config(collapsed, *memory_config);
    }

    return {post, comb, collapsed};
}

}  // namespace ttnn::experimental::deepseek::hyperconnection
