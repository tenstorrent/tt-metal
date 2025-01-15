// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fill_pad.hpp"
#include "device/fill_pad_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/core.hpp"
#include <utility>

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

ttnn::Tensor FillPadOperation::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    float fill_value,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    auto output_memory_config = memory_config.value_or(input_tensor.memory_config());
    // if input_tensor is rank > 3, then we need to reshape it to rank 3 such that the last 2 dims are the same
    if (input_tensor.get_logical_shape().rank() > 3) {
        ttnn::SimpleShape original_shape = input_tensor.get_logical_shape();
        ttnn::SimpleShape new_shape = ttnn::SimpleShape{
            std::array<uint32_t, 4>{1, original_shape[0] * original_shape[1], original_shape[2], original_shape[3]}};
        auto reshaped_tensor = ttnn::reshape(input_tensor, new_shape);
        operation::run_without_autoformat(
            FillPad{fill_value, output_memory_config}, {reshaped_tensor}, {}, {}, queue_id)
            .at(0);
        return ttnn::reshape(reshaped_tensor, original_shape);
    }
    return operation::run_without_autoformat(
               FillPad{fill_value, output_memory_config}, {input_tensor}, {}, {}, queue_id)
        .at(0);
}

ttnn::Tensor FillPadOperation::invoke(
    const ttnn::Tensor& input_tensor, float fill_value, const std::optional<ttnn::MemoryConfig>& memory_config_arg) {
    return invoke(DefaultQueueId, input_tensor, fill_value, memory_config_arg);
}

}  // namespace ttnn::operations::data_movement
