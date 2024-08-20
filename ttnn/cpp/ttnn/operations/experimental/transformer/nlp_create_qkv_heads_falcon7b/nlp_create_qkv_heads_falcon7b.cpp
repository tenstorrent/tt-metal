// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_falcon7b.hpp"

namespace ttnn::operations::experimental::transformer {

    std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> NLPCreateHeadsFalcon7bOperation::invoke (
        uint8_t queue_id,
        const Tensor& input_tensor_q,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<std::vector<std::optional<Tensor>>> optional_output_tensors) {
            const MemoryConfig output_mem_config = memory_config.value_or(input_tensor_q.memory_config());
            auto optional_outputs = std::vector<std::optional<Tensor>>{};
            if (optional_output_tensors.has_value()) {
                optional_outputs = {optional_output_tensors.value().begin(), optional_output_tensors.value().end()};
            }
            else {
                optional_outputs = {};
            }
            auto outputs = operation::run(NlpCreateHeadsFalcon7BDeviceOperation{output_mem_config}, {input_tensor_q}, {}, optional_outputs);
            return {outputs[0], outputs[1], outputs[2]};
    };

    std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> NLPCreateHeadsFalcon7bOperation::invoke (
        const Tensor& input_tensor_q,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<std::vector<std::optional<Tensor>>> optional_output_tensors) {
        return invoke(ttnn::DefaultQueueId, input_tensor_q, memory_config, optional_output_tensors);
    };

}  // namespace ttnn::operations::experimental::transformer
