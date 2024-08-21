// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize.hpp"

#include "device/tilize_op.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::data_movement {

ttnn::Tensor ExecuteTilize::invoke(
    uint8_t queue_id,
    const ttnn::Tensor &input_tensor,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<DataType> output_dtype,
    bool use_multicore) {
    return operation::run(
               Tilize{
                   memory_config.value_or(input_tensor.memory_config()),
                   output_dtype.value_or(input_tensor.get_dtype()),
                   use_multicore},
               {input_tensor},
               {},
               {},
               queue_id)
        .at(0);
}

ttnn::Tensor ExecuteTilize::invoke(
    const ttnn::Tensor &input_tensor,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<DataType> output_dtype,
    bool use_multicore) {
    return invoke(DefaultQueueId, input_tensor, memory_config, output_dtype, use_multicore);
}

}  // namespace ttnn::operations::data_movement
