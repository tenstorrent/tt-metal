// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/nlp_create_qkv_heads_boltz_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {
namespace operations::experimental::transformer {
struct NlpCreateHeadsBoltzOperation {
    static std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> invoke(
        const Tensor& input_tensor_q,
        const std::optional<Tensor>& input_tensor_kv,
        uint32_t num_q_heads,
        std::optional<uint32_t> num_kv_heads,
        bool transpose_k_heads,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<std::vector<std::optional<Tensor>>>& optional_output_tensors = std::nullopt);
};
}  // namespace operations::experimental::transformer

namespace experimental {

constexpr auto nlp_create_qkv_heads_boltz = ttnn::register_operation<
    "ttnn::experimental::nlp_create_qkv_heads_boltz",
    ttnn::operations::experimental::transformer::NlpCreateHeadsBoltzOperation>();

}  // namespace experimental
}  // namespace ttnn
