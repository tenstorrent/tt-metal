// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_boltz.hpp"

#include <utility>
namespace ttnn::experimental {
std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> nlp_create_qkv_heads_boltz(
    const Tensor& input_tensor_q,
    const std::optional<Tensor>& input_tensor_kv,
    const uint32_t num_q_heads,
    const std::optional<uint32_t> num_kv_heads,
    const bool transpose_k_heads,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<std::vector<std::optional<Tensor>>>& optional_output_tensors) {
    TT_FATAL(num_q_heads > 0, "num_q_heads must be greater than 0");
    const uint32_t num_kv_heads_val = num_kv_heads.value_or(num_q_heads);
    // Every head takes at least one column, so bounding the head counts by the width they split also keeps
    // the uint32_t section sums below from wrapping to 0.
    const uint64_t q_width = input_tensor_q.padded_shape()[3];
    uint32_t head_dim;
    if (input_tensor_kv.has_value()) {
        TT_FATAL(num_kv_heads_val > 0, "num_kv_heads must be greater than 0 when a KV tensor is given");
        TT_FATAL(num_q_heads <= q_width, "num_q_heads ({}) exceeds the Q width ({})", num_q_heads, q_width);
        TT_FATAL(
            2 * uint64_t{num_kv_heads_val} <= input_tensor_kv.value().padded_shape()[3],
            "2 * num_kv_heads ({}) exceeds the KV width ({})",
            num_kv_heads_val,
            input_tensor_kv.value().padded_shape()[3]);
        TT_FATAL(input_tensor_q.padded_shape()[3] % num_q_heads == 0, "Unsupported input shape");
        TT_FATAL(input_tensor_kv.value().padded_shape()[3] % (2 * num_kv_heads_val) == 0, "Unsupported input shape");
        head_dim = input_tensor_q.padded_shape()[3] / num_q_heads;
        TT_FATAL(
            input_tensor_kv.value().padded_shape()[3] / (2 * num_kv_heads_val) == head_dim,
            "Head dims must be the same for Q and K, V");
    } else {
        TT_FATAL(
            uint64_t{num_q_heads} + 2 * uint64_t{num_kv_heads_val} <= q_width,
            "num_q_heads ({}) + 2 * num_kv_heads ({}) exceeds the fused width ({})",
            num_q_heads,
            num_kv_heads_val,
            q_width);
        TT_FATAL(
            input_tensor_q.padded_shape()[3] % (num_q_heads + 2 * num_kv_heads_val) == 0, "Unsupported input shape");
        head_dim = input_tensor_q.padded_shape()[3] / (num_q_heads + 2 * num_kv_heads_val);
    }

    return ttnn::prim::nlp_create_qkv_heads_boltz(
        input_tensor_q,
        input_tensor_kv,
        num_q_heads,
        num_kv_heads,
        head_dim,
        transpose_k_heads,
        memory_config,
        optional_output_tensors);
}

}  // namespace ttnn::experimental
