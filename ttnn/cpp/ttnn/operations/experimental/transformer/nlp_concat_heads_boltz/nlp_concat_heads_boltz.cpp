// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/nlp_concat_heads_boltz_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "nlp_concat_heads_boltz.hpp"

#include <utility>

namespace ttnn::operations::experimental::transformer {

ttnn::Tensor NLPConcatHeadsBoltzOperation::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    return tt::tt_metal::operation::run(
               NLPConcatHeadsBoltzDeviceOperation{memory_config.value_or(input_tensor.memory_config())},
               {input_tensor},
               {},
               {std::move(optional_output_tensor)})
        .at(0);
}

ttnn::Tensor NLPConcatHeadsBoltzOperation::invoke(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    return invoke(ttnn::DefaultQueueId, input_tensor, memory_config, std::move(optional_output_tensor));
}
};  // namespace ttnn::operations::experimental::transformer
