// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "non_zero_indices.hpp"
#include "device/non_zero_indices_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/common/constants.hpp"

namespace ttnn::operations::data_movement {

std::vector<ttnn::Tensor> NonZeroIndicesOperation::invoke(uint8_t queue_id,
                                                          const ttnn::Tensor& input_tensor,
                                                          const std::optional<MemoryConfig>& memory_config_arg) {
    auto memory_config = memory_config_arg.value_or(input_tensor.memory_config());
    return operation::run_without_autoformat(NonZeroIndices{memory_config}, {input_tensor}, {}, {}, queue_id);
}

std::vector<ttnn::Tensor> NonZeroIndicesOperation::invoke(const ttnn::Tensor& input_tensor,
                                                          const std::optional<MemoryConfig>& memory_config_arg) {
    return invoke(DefaultQueueId, input_tensor, memory_config_arg);
}

}  // namespace ttnn::operations::data_movement
