// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "create_qkv_heads.hpp"

#include <utility>
#include "device/create_qkv_heads_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::experimental::create_qkv_heads {

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> CreateQKVHeadsOperation::invoke(
    const Tensor& input_tensor,
    const uint32_t num_q_heads,
    const std::optional<uint32_t> num_kv_heads,
    const bool transpose_k_heads,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<std::array<Tensor, 3>> optional_output_tensors) {
    const MemoryConfig output_mem_config = memory_config.value_or(input_tensor.memory_config());
    const uint32_t num_kv_heads_val = num_kv_heads.value_or(num_q_heads);
    TT_FATAL(
        input_tensor.padded_shape()[3] % (num_q_heads + (2 * num_kv_heads_val)) == 0,
        "Flattened hidden dimension {} must be a multiple of the combined Q {}, K {} and V {} heads",
        input_tensor.padded_shape()[3],
        num_q_heads,
        num_kv_heads_val,
        num_kv_heads_val);
    const uint32_t head_dim = input_tensor.padded_shape()[3] / (num_q_heads + (2 * num_kv_heads_val));

    std::optional<std::tuple<Tensor, Tensor, Tensor>> preallocated_outputs = std::nullopt;
    if (optional_output_tensors.has_value()) {
        const auto& arr = optional_output_tensors.value();
        preallocated_outputs = std::make_tuple(arr[0], arr[1], arr[2]);
    }

    return ttnn::prim::create_qkv_heads(
        input_tensor, num_q_heads, num_kv_heads_val, head_dim, transpose_k_heads, memory_config, preallocated_outputs);
}

}  // namespace ttnn::operations::experimental::create_qkv_heads
