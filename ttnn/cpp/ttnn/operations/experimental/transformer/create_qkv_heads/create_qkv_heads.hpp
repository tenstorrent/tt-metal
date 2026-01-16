// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/create_qkv_heads_device_operation.hpp"
#include "ttnn/operation.hpp"

namespace ttnn {
namespace operations::experimental::create_qkv_heads {

struct CreateQKVHeadsOperation {
    static std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> invoke(
        const Tensor& input_tensor,
        uint32_t num_q_heads,
        std::optional<uint32_t> num_kv_heads,
        bool transpose_k_heads,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<std::array<Tensor, 3>> optional_output_tensors = std::nullopt);
};

}  // namespace operations::experimental::create_qkv_heads

namespace experimental {

constexpr auto create_qkv_heads = ttnn::register_operation<
    "ttnn::experimental::create_qkv_heads",
    ttnn::operations::experimental::create_qkv_heads::CreateQKVHeadsOperation>();

}  // namespace experimental

}  // namespace ttnn
