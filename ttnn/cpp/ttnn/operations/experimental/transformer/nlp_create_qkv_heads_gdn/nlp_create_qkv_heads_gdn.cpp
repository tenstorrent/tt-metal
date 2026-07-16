// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_gdn.hpp"

namespace ttnn::experimental {

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> nlp_create_qkv_heads_gdn(
    const Tensor& input_tensor,
    const uint32_t num_q_heads,
    const uint32_t num_k_heads,
    const uint32_t num_v_heads,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<std::vector<std::optional<Tensor>>>& optional_output_tensors) {
    const uint32_t total_heads = num_q_heads + num_k_heads + num_v_heads;
    TT_FATAL(
        input_tensor.padded_shape()[3] % total_heads == 0,
        "Fused width {} must be divisible by (Nq+Nk+Nv) = {}",
        input_tensor.padded_shape()[3],
        total_heads);
    const uint32_t head_dim = input_tensor.padded_shape()[3] / total_heads;

    return ttnn::prim::nlp_create_qkv_heads_gdn(
        input_tensor, num_q_heads, num_k_heads, num_v_heads, head_dim, memory_config, optional_output_tensors);
}

}  // namespace ttnn::experimental
