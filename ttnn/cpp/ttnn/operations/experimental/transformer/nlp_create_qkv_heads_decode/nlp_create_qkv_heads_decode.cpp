// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_decode.hpp"
#include "device/nlp_create_qkv_heads_decode_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/common/constants.hpp"

namespace ttnn::operations::experimental::transformer {

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> NLPCreateHeadsDecodeOperation::invoke(
    uint8_t queue_id,
    const Tensor &input_tensor,
    const uint32_t num_heads,
    const std::optional<const uint32_t> num_kv_heads,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<std::array<Tensor, 3>> optional_output_tensors) {
    const uint32_t num_kv_heads_val = num_kv_heads.value_or(num_heads);
    // Infer head_dim
    TT_FATAL(input_tensor.get_legacy_shape()[3] % (num_heads + 2 * num_kv_heads_val) == 0, "Unsupported input shape");
    uint32_t head_dim = input_tensor.get_legacy_shape()[3] / (num_heads + 2 * num_kv_heads_val);
    auto optional_outputs = std::vector<std::optional<Tensor>>{};
    if (optional_output_tensors.has_value()) {
        optional_outputs = {optional_output_tensors.value().begin(), optional_output_tensors.value().end()};
    } else {
        optional_outputs = {};
    }
    auto out = operation::run(
        NLPCreateHeadsDecodeDeviceOperation{
            num_heads, num_kv_heads_val, head_dim, memory_config.value_or(input_tensor.memory_config())},
        {input_tensor},
        {},
        optional_outputs,
        queue_id);
    return {out.at(0), out.at(1), out.at(2)};
}

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> NLPCreateHeadsDecodeOperation::invoke(
    const Tensor &input_tensor,
    const uint32_t num_heads,
    const std::optional<const uint32_t> num_kv_heads,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<std::array<Tensor, 3>> optional_output_tensors) {
    return invoke(ttnn::DefaultQueueId, input_tensor, num_heads, num_kv_heads, memory_config, optional_output_tensors);
}

}  // namespace ttnn::operations::experimental::transformer
