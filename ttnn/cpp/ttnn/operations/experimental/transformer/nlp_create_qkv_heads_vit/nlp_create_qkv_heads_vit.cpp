// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_vit.hpp"
#include "ttnn/graph/composite_trace.hpp"

namespace ttnn::experimental {

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> nlp_create_qkv_heads_vit(
    const Tensor& input_tensor_q,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<std::vector<std::optional<Tensor>>>& optional_output_tensors) {
    TT_OP_SCOPE("ttnn::experimental::nlp_create_qkv_heads_vit");
    const MemoryConfig output_mem_config = memory_config.value_or(input_tensor_q.memory_config());

    auto outputs = ttnn::prim::nlp_create_qkv_heads_vit(input_tensor_q, output_mem_config, optional_output_tensors);
    return {outputs[0], outputs[1], outputs[2]};
}

}  // namespace ttnn::experimental
