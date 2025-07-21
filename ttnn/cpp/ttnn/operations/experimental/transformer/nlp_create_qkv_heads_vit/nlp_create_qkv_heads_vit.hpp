// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/nlp_create_qkv_heads_vit_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {
namespace operations::experimental::transformer {

struct NLPCreateHeadsVitOperation {
    static std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> invoke(
        QueueId queue_id,
        const Tensor& input_tensor_q,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<std::vector<std::optional<Tensor>>> optional_output_tensors = std::nullopt);

    static std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> invoke(
        const Tensor& input_tensor_q,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<std::vector<std::optional<ttnn::Tensor>>> optional_output_tensors = std::nullopt);
};
}  // namespace operations::experimental::transformer

namespace experimental {

constexpr auto nlp_create_qkv_heads_vit = ttnn::register_operation<
    "ttnn::experimental::nlp_create_qkv_heads_vit",
    ttnn::operations::experimental::transformer::NLPCreateHeadsVitOperation>();

}  // namespace experimental

}  // namespace ttnn
