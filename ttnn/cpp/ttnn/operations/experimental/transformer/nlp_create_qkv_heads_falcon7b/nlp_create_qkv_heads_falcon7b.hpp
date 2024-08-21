// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/nlp_create_qkv_heads_falcon7b_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {
namespace operations::experimental::transformer {

struct NLPCreateHeadsFalcon7bOperation {
    static std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> invoke (
        uint8_t queue_id,
        const Tensor& input_tensor_q,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<std::vector<std::optional<Tensor>>> optional_output_tensors = std::nullopt);

    static std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> invoke (
        const Tensor& input_tensor_q,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<std::vector<std::optional<ttnn::Tensor>>> optional_output_tensors = std::nullopt);
};
} // namespace operations::experimental::transformer


namespace experimental {

constexpr auto nlp_create_qkv_heads_falcon7b = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::nlp_create_qkv_heads_falcon7b",
    ttnn::operations::experimental::transformer::NLPCreateHeadsFalcon7bOperation>();

}  // namespace experimental

}  // namespace ttnn
