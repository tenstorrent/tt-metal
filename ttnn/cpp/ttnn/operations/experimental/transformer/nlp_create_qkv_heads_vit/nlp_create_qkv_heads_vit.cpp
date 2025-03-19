// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_vit.hpp"

#include <utility>

namespace ttnn::operations::experimental::transformer {

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> NLPCreateHeadsVitOperation::invoke(
    QueueId queue_id,
    const Tensor& input_tensor_q,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<std::vector<std::optional<Tensor>>> optional_output_tensors) {
    const MemoryConfig output_mem_config = memory_config.value_or(input_tensor_q.memory_config());
    auto optional_outputs = std::vector<std::optional<Tensor>>{};
    if (optional_output_tensors.has_value()) {
        optional_outputs = {optional_output_tensors.value().begin(), optional_output_tensors.value().end()};
    } else {
        optional_outputs = {};
    }
    auto outputs = tt::tt_metal::operation::run(
        NlpCreateHeadsVitDeviceOperation{output_mem_config}, {input_tensor_q}, {}, optional_outputs);
    return {outputs[0], outputs[1], outputs[2]};
};

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> NLPCreateHeadsVitOperation::invoke(
    const Tensor& input_tensor_q,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<std::vector<std::optional<Tensor>>> optional_output_tensors) {
    return invoke(ttnn::DefaultQueueId, input_tensor_q, memory_config, std::move(optional_output_tensors));
};

}  // namespace ttnn::operations::experimental::transformer
