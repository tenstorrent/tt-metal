// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deltanet_decode_full.hpp"
#include "device/deltanet_full_device_operation.hpp"

namespace ttnn::experimental {

std::vector<Tensor> deltanet_decode_full(
    const Tensor& qkv_proj,
    const Tensor& z_proj,
    const Tensor& b_proj,
    const Tensor& a_proj,
    const Tensor& conv_state,
    const Tensor& recurrent_state,
    const Tensor& conv1d_weight,
    const Tensor& a_log,
    const Tensor& dt_bias,
    const Tensor& norm_weight,
    uint32_t num_heads,
    uint32_t num_k_heads,
    uint32_t k_head_dim,
    uint32_t v_head_dim,
    uint32_t conv_dim,
    uint32_t conv_kernel_size,
    uint32_t head_expand_ratio,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::deltanet_decode_full(
        qkv_proj, z_proj, b_proj, a_proj,
        conv_state, recurrent_state,
        conv1d_weight, a_log, dt_bias, norm_weight,
        num_heads, num_k_heads, k_head_dim, v_head_dim,
        conv_dim, conv_kernel_size, head_expand_ratio,
        memory_config);
}

}  // namespace ttnn::experimental
