// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/create_qkv_heads_from_separate_tensors_device_operation.hpp"
#include "ttnn/operation.hpp"

namespace ttnn {
namespace operations::experimental::transformer {

struct CreateQKVHeadsSeparateTensorsOperation {
    static std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> invoke(
        const Tensor& input_tensor,
        const Tensor& input_tensor_kv,
        uint32_t num_q_heads,
        std::optional<uint32_t> num_kv_heads,
        bool transpose_k_heads,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<std::array<Tensor, 3>>& optional_output_tensors = std::nullopt);
};

}  // namespace operations::experimental::transformer

namespace experimental {

constexpr auto create_qkv_heads_from_separate_tensors = ttnn::register_operation<
    "ttnn::experimental::create_qkv_heads_from_separate_tensors",
    ttnn::operations::experimental::transformer::CreateQKVHeadsSeparateTensorsOperation>();

}  // namespace experimental

}  // namespace ttnn
