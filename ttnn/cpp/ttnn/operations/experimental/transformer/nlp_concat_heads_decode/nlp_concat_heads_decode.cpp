// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/nlp_concat_heads_decode_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "nlp_concat_heads_decode.hpp"

namespace ttnn::operations::experimental::transformer {

    ttnn::Tensor NLPConcatHeadsDecodeOperation::invoke(
        uint8_t queue_id,
        const Tensor& input_tensor,
        const uint32_t num_heads,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<Tensor> optional_output_tensor) {
        return operation::run(NLPConcatHeadsDecodeDeviceOperation{num_heads}, {input_tensor}, {}, {optional_output_tensor}).at(0);
    }

    ttnn::Tensor NLPConcatHeadsDecodeOperation::invoke(
        const Tensor& input_tensor,
        const uint32_t num_heads,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<Tensor> optional_output_tensor) {
        return invoke(
            ttnn::DefaultQueueId, input_tensor, num_heads, memory_config, optional_output_tensor);
    }
};  // namespace ttnn::operations::experimental::transformer
