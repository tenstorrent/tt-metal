// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding.hpp"

#include "device/untilize_with_unpadding_op.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::data_movement {

ttnn::Tensor ExecuteUntilizeWithUnpadding::invoke(
    uint8_t queue_id,
    const ttnn::Tensor &input_tensor,
    const tt::tt_metal::LegacyShape &output_tensor_end,
    const std::optional<MemoryConfig> &memory_config,
    bool use_multicore,
    bool use_pack_untilize) {
    // MT: Currently only uint32 is moved to DST directly, fp32 is converted to fp16b
    bool fp32_dest_acc_en = input_tensor.get_dtype() == DataType::UINT32;

    return operation::run(
               UntilizeWithUnpadding{
                   output_tensor_end,
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

ttnn::Tensor ExecuteUntilizeWithUnpadding::invoke(
    const ttnn::Tensor &input_tensor,
    const tt::tt_metal::LegacyShape &output_tensor_end,
    const std::optional<MemoryConfig> &memory_config,
    bool use_multicore,
    bool use_pack_untilize) {
    return invoke(DefaultQueueId, input_tensor, output_tensor_end, memory_config, use_multicore, use_pack_untilize);
}

}  // namespace ttnn::operations::data_movement
