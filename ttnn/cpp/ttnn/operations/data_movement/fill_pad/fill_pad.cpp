// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fill_pad.hpp"
#include "device/fill_pad_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/common/constants.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

ttnn::Tensor FillPadOperation::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    float fill_value,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    auto output_memory_config = memory_config.value_or(input_tensor.memory_config());
    return operation::run_without_autoformat(
               FillPad{fill_value, output_memory_config}, {input_tensor}, {}, {}, queue_id)
        .at(0);
}

ttnn::Tensor FillPadOperation::invoke(
    const ttnn::Tensor& input_tensor, float fill_value, const std::optional<ttnn::MemoryConfig>& memory_config_arg) {
    return invoke(DefaultQueueId, input_tensor, fill_value, memory_config_arg);
}

}  // namespace ttnn::operations::data_movement
