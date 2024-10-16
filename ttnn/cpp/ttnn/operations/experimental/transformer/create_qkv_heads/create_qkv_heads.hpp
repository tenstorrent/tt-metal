// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/create_qkv_heads_device_operation.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn {
namespace operations::experimental::transformer {

struct CreateQKVHeadsOperation {
    static std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> invoke(
        uint8_t queue_id,
        const Tensor &input_tensor,
        const uint32_t num_q_heads,
        const std::optional<uint32_t> num_kv_heads,
        const bool transpose_k_heads,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<std::array<Tensor, 3>> optional_output_tensors = std::nullopt);

    static std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> invoke(
        const Tensor &input_tensor,
        const uint32_t num_q_heads,
        const std::optional<uint32_t> num_kv_heads,
        const bool transpose_k_heads,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<std::array<Tensor, 3>> optional_output_tensors = std::nullopt);
};

}  // namespace operations::experimental::transformer

namespace experimental {

constexpr auto create_qkv_heads = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::create_qkv_heads",
    ttnn::operations::experimental::transformer::CreateQKVHeadsOperation>();

}  // namespace experimental

}  // namespace ttnn
