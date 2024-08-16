// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize.hpp"

#include "device/untilize_op.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::data_movement {

ttnn::Tensor ExecuteUntilize::invoke(
    uint8_t queue_id,
    const ttnn::Tensor &input_tensor,
    const std::optional<MemoryConfig> &memory_config,
    bool use_multicore,
    bool use_pack_untilize) {
    bool fp32_dest_acc_en =
        input_tensor.get_dtype() ==
        DataType::UINT32;  // MT: Currently only uint32 is moved to DST directly, fp32 is converted to fp16b

    return operation::run(
               Untilize{
                   memory_config.value_or(input_tensor.memory_config()),
                   use_multicore,
                   use_pack_untilize,
                   fp32_dest_acc_en},
               {input_tensor},
               {},
               {},
               queue_id)
        .at(0);
}

ttnn::Tensor ExecuteUntilize::invoke(
    const ttnn::Tensor &input_tensor,
    const std::optional<MemoryConfig> &memory_config,
    bool use_multicore,
    bool use_pack_untilize) {
    return invoke(DefaultQueueId, input_tensor, memory_config, use_multicore, use_pack_untilize);
}

}  // namespace ttnn::operations::data_movement
