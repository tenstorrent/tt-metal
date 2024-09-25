// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "create_qkv_heads_from_separate_tensors.hpp"
#include "device/create_qkv_heads_from_separate_tensors_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/common/constants.hpp"

namespace ttnn:: operations::experimental::transformer {

    std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> CreateQKVHeadsSeparateTensorsOperation::invoke(
        uint8_t queue_id,
        const Tensor &input_tensor_q,
        const Tensor &input_tensor_kv,
        const uint32_t num_q_heads,
        const std::optional<uint32_t> num_kv_heads,
        const bool transpose_k_heads,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<std::array<Tensor, 3>> optional_output_tensors) {

        const uint32_t num_kv_heads_val = num_kv_heads.value_or(num_q_heads);
        TT_FATAL(input_tensor_q.get_shape().with_tile_padding()[3] % num_q_heads == 0, "Flattened Q hidden dimension {} must be a multiple of Q heads {}", input_tensor_q.get_shape().with_tile_padding()[3], num_q_heads);
        const uint32_t head_dim = input_tensor_q.get_shape().with_tile_padding()[3] / (num_q_heads);
        auto optional_outputs = std::vector<std::optional<Tensor>>{};
        if (optional_output_tensors.has_value()) {
            optional_outputs = {optional_output_tensors.value().begin(), optional_output_tensors.value().end()};
        }
        else {
            optional_outputs = {};
        }
        auto output_tensors = operation::run(CreateQKVHeadsSeparateTensorsDeviceOperation{num_q_heads, num_kv_heads_val, head_dim, transpose_k_heads, memory_config.value_or(input_tensor_q.memory_config())}, {input_tensor_q, input_tensor_kv}, {}, optional_outputs, queue_id);
        return {output_tensors.at(0), output_tensors.at(1), output_tensors.at(2)};
    }

    std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> CreateQKVHeadsSeparateTensorsOperation::invoke(
        const Tensor &input_tensor_q,
        const Tensor &input_tensor_kv,
        const uint32_t num_q_heads,
        const std::optional<uint32_t> num_kv_heads,
        const bool transpose_k_heads,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<std::array<Tensor, 3>> optional_output_tensors) {
        return invoke(
            ttnn::DefaultQueueId, input_tensor_q, input_tensor_kv, num_q_heads, num_kv_heads, transpose_k_heads, memory_config, optional_output_tensors);
    }

}  // namespace ttnn::operations::experimental::transformer
