// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads.hpp"

namespace ttnn::operations::experimental::transformer {

    std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> NlpCreateHeadsOperation::invoke (
        uint8_t queue_id,
        const Tensor& input_tensor_q,
        const std::optional<Tensor>& input_tensor_kv,
        const uint32_t num_q_heads,
        const std::optional<uint32_t> num_kv_heads,
        const bool transpose_k_heads,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<std::vector<std::optional<Tensor>>> optional_output_tensors) {
            const uint32_t num_kv_heads_val = num_kv_heads.value_or(num_q_heads);
            uint32_t head_dim;
            if (input_tensor_kv.has_value()) {
                TT_FATAL(input_tensor_q.get_legacy_shape()[3] % num_q_heads == 0, "Unsupported input shape");
                TT_FATAL(input_tensor_kv.value().get_legacy_shape()[3] % (2 * num_kv_heads_val) == 0, "Unsupported input shape");
                head_dim = input_tensor_q.get_legacy_shape()[3] / num_q_heads;
                TT_FATAL(input_tensor_kv.value().get_legacy_shape()[3] / (2 * num_kv_heads_val) == head_dim, "Head dims must be the same for Q and K, V");
            } else {
                TT_FATAL(input_tensor_q.get_legacy_shape()[3] % (num_q_heads + 2 * num_kv_heads_val) == 0, "Unsupported input shape");
                head_dim = input_tensor_q.get_legacy_shape()[3] / (num_q_heads + 2 * num_kv_heads_val);
            }

            return ttnn::prim::nlp_create_qkv_heads(
                queue_id,
                {input_tensor_q, input_tensor_kv, optional_output_tensors.value_or(std::vector<std::optional<Tensor>>{})},
                {num_q_heads, num_kv_heads.value_or(num_q_heads), head_dim, transpose_k_heads, memory_config.value_or(input_tensor_q.memory_config())});
    };

    std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> NlpCreateHeadsOperation::invoke (
        const Tensor& input_tensor_q,
        const std::optional<Tensor>& input_tensor_kv,
        const uint32_t num_q_heads,
        const std::optional<uint32_t> num_kv_heads,
        const bool transpose_k_heads,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<std::vector<std::optional<ttnn::Tensor>>> optional_output_tensors) {
        return invoke(ttnn::DefaultQueueId, input_tensor_q, input_tensor_kv, num_q_heads, num_kv_heads, transpose_k_heads, memory_config, optional_output_tensors);
    };

}  // namespace ttnn::operations::experimental::transformer
