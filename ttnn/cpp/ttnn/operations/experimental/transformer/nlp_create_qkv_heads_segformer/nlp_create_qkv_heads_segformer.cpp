// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_segformer/nlp_create_qkv_heads_segformer.hpp"
#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_segformer/device/nlp_create_qkv_heads_segformer_device_operation.hpp"

namespace ttnn::operations::experimental::transformer {

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> NLPCreateHeadsSegformerOperation::invoke(
    const Tensor& input_tensor_q,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<std::vector<std::optional<Tensor>>>& optional_output_tensors) {
    const MemoryConfig output_mem_config = memory_config.value_or(input_tensor_q.memory_config());
    std::vector<std::optional<Tensor>> output_tensors;
    if (optional_output_tensors.has_value()) {
        output_tensors = optional_output_tensors.value();
    }
    return ttnn::prim::nlp_create_qkv_heads_segformer(input_tensor_q, output_mem_config, output_tensors);
}

}  // namespace ttnn::operations::experimental::transformer
