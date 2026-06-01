// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_hyperconnection.hpp"

#include "device/fused_pre_post_device_operation.hpp"
#include "device/sinkhorn_device_operation.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"

namespace ttnn::experimental::deepseek::hyperconnection {

std::tuple<Tensor, Tensor, Tensor> fused_hyperconnection(
    const Tensor& hidden_streams,
    const Tensor& pre_w,
    const Tensor& post_w,
    const Tensor& comb_w,
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

    // Decode-only fused stage (T == 1):
    //   post      = 2 * sigmoid(post_w * post_scale + post_bias)            [1,1,1,H]
    //   collapsed = (sigmoid(pre_w * pre_scale + pre_bias) + eps) @ hidden  [1,1,1,D]
    // The pre-weighted stream collapse is fused into the device op as a [1,H] x [H,D] matmul.
    auto [post, collapsed] = ttnn::prim::fused_hyperconnection_pre_post(
        pre_w, post_w, pre_bias, post_bias, hidden_streams, pre_scale, post_scale, eps, memory_config);

    // comb: softmax(comb_w * comb_scale + comb_bias, dim=-1) + eps, then Sinkhorn (alternate
    // row/col normalisation) onto the doubly-stochastic manifold, fused into a single device op.
    // The [1,1,1,H*H] projection/bias rows are reshaped to the [1,1,H,H] comb matrix; the device
    // op masks the valid HxH block inside the 32x32 tile.
    Tensor comb_w_mat = ttnn::reshape(comb_w, ttnn::Shape({1, 1, hc, hc}));
    Tensor comb_bias_mat = ttnn::reshape(comb_bias, ttnn::Shape({1, 1, hc, hc}));
    Tensor comb = ttnn::prim::fused_hyperconnection_sinkhorn(
        comb_w_mat, comb_bias_mat, hc, sinkhorn_iters, comb_scale, eps, memory_config);

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
