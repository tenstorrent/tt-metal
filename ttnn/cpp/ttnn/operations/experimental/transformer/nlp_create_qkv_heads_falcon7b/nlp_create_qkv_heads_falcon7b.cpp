// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_falcon7b.hpp"

#include <utility>

namespace ttnn::operations::experimental::transformer {

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> NLPCreateHeadsFalcon7bOperation::invoke(
    const Tensor& input_tensor_q, const std::optional<MemoryConfig>& memory_config) {
    auto result = ttnn::prim::nlp_create_qkv_heads_falcon7b(input_tensor_q, memory_config);
    return {result.q, result.k, result.v};
}

}  // namespace ttnn::operations::experimental::transformer
