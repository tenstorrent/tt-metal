// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_vit.hpp"

namespace ttnn::operations::experimental::transformer {

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> NLPCreateHeadsVitOperation::invoke(
    const Tensor& input_tensor_q,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<std::vector<std::optional<Tensor>>>& optional_output_tensors) {
    const MemoryConfig output_mem_config = memory_config.value_or(input_tensor_q.memory_config());

    auto outputs = ttnn::prim::nlp_create_qkv_heads_vit(input_tensor_q, output_mem_config, optional_output_tensors);
    return {outputs[0], outputs[1], outputs[2]};
}

}  // namespace ttnn::operations::experimental::transformer
