// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::experimental::transformer {

struct NLPCreateHeadsSegformerOperation {
    static std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> invoke(
        const Tensor& input_tensor_q,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<std::vector<std::optional<Tensor>>>& optional_output_tensors = std::nullopt);
};
}  // namespace operations::experimental::transformer

namespace experimental {

constexpr auto nlp_create_qkv_heads_segformer = ttnn::register_operation<
    "ttnn::experimental::nlp_create_qkv_heads_segformer",
    ttnn::operations::experimental::transformer::NLPCreateHeadsSegformerOperation>();

}  // namespace experimental

}  // namespace ttnn
