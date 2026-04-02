// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "deltanet.hpp"
#include "device/deltanet_device_operation.hpp"

namespace ttnn::experimental {

Tensor deltanet_recurrence(
    const Tensor& conv_out,
    const Tensor& b_proj,
    const Tensor& a_proj,
    const Tensor& z_proj,
    const Tensor& dt_bias,
    const Tensor& A_exp,
    const Tensor& norm_weight,
    const Tensor& state,
    uint32_t num_heads,
    uint32_t head_k_dim,
    uint32_t head_v_dim,
    uint32_t num_k_heads,
    uint32_t gqa_ratio,
    float scale,
    float norm_eps) {
    return ttnn::prim::deltanet_recurrence(
        conv_out,
        b_proj,
        a_proj,
        z_proj,
        dt_bias,
        A_exp,
        norm_weight,
        state,
        num_heads,
        head_k_dim,
        head_v_dim,
        num_k_heads,
        gqa_ratio,
        scale,
        norm_eps);
}

}  // namespace ttnn::experimental
