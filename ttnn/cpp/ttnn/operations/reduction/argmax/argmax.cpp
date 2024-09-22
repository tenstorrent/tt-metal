// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/argmax_op.hpp"
#include "ttnn/operations/reduction/argmax/argmax.hpp"

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::reduction {

ttnn::Tensor ArgMaxOperation::invoke(
    uint8_t queue_id,
    const Tensor& input_tensor,
    const std::optional<int> dim,
    const bool use_muticore,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    return operation::run(
                ArgMax{tt::tt_metal::DataType::UINT32, dim, use_muticore, memory_config.value_or(input_tensor.memory_config())},
                {input_tensor}, {}, {optional_output_tensor}, queue_id)
        .at(0);
}

ttnn::Tensor ArgMaxOperation::invoke(
    const Tensor& input_tensor,
    const std::optional<int> dim,
    const bool use_muticore,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    return invoke(DefaultQueueId, input_tensor, dim, use_muticore, memory_config, optional_output_tensor);
}


}  // namespace ttnn::operations::reduction
