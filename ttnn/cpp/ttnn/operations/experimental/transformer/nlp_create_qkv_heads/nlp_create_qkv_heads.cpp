// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads.hpp"

#include <utility>

namespace ttnn::experimental {

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> nlp_create_qkv_heads(
    const Tensor& input_tensor_q,
    const std::optional<Tensor>& input_tensor_kv,
    const uint32_t num_q_heads,
    const std::optional<uint32_t> num_kv_heads,
    const bool transpose_k_heads,
    const bool kv_tied,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<std::vector<std::optional<Tensor>>>& optional_output_tensors) {
    const uint32_t num_kv_heads_val = num_kv_heads.value_or(num_q_heads);
    uint32_t head_dim;
    if (input_tensor_kv.has_value()) {
        // Tied, so the KV tensor holds one K/V section rather than two.
        const uint32_t kv_sections = kv_tied ? 1 : 2;
        TT_FATAL(input_tensor_q.padded_shape()[3] % num_q_heads == 0, "Unsupported input shape");
        TT_FATAL(
            input_tensor_kv.value().padded_shape()[3] % (kv_sections * num_kv_heads_val) == 0,
            "Unsupported input shape");
        head_dim = input_tensor_q.padded_shape()[3] / num_q_heads;
        TT_FATAL(
            input_tensor_kv.value().padded_shape()[3] / (kv_sections * num_kv_heads_val) == head_dim,
            "Head dims must be the same for Q and K, V");
    } else if (kv_tied) {
        // One K/V section instead of two, so the fused width is (q + kv), not (q + 2*kv).
        const uint32_t fused_width = input_tensor_q.padded_shape()[3];
        const uint32_t tied_sections = num_q_heads + num_kv_heads_val;
        TT_FATAL(fused_width % tied_sections == 0, "Unsupported input shape");
        const uint32_t untied_sections = num_q_heads + 2 * num_kv_heads_val;
        TT_FATAL(
            fused_width % untied_sections != 0,
            "Ambiguous kv_tied fused input shape: width {} is divisible by both {} (tied) and {} (untied) sections",
            fused_width,
            tied_sections,
            untied_sections);
        head_dim = fused_width / tied_sections;
    } else {
        TT_FATAL(
            input_tensor_q.padded_shape()[3] % (num_q_heads + 2 * num_kv_heads_val) == 0, "Unsupported input shape");
        head_dim = input_tensor_q.padded_shape()[3] / (num_q_heads + 2 * num_kv_heads_val);
    }

    return ttnn::prim::nlp_create_qkv_heads(
        input_tensor_q,
        input_tensor_kv,
        num_q_heads,
        num_kv_heads,
        head_dim,
        transpose_k_heads,
        kv_tied,
        memory_config,
        optional_output_tensors);
}

}  // namespace ttnn::experimental
