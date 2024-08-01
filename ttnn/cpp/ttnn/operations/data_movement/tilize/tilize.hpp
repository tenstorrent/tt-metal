// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "device/tilize_op.hpp"

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/common.hpp"

namespace ttnn {
namespace operations::data_movement {

struct ExecuteTilize {
    static ttnn::Tensor operator()(
        uint8_t queue_id,
        const ttnn::Tensor &input_tensor,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<DataType> output_dtype = std::nullopt,
        bool use_multicore = false) {
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

    static ttnn::Tensor operator()(
        const ttnn::Tensor &input_tensor,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<DataType> output_dtype = std::nullopt,
        bool use_multicore = false) {
        return operator()(DefaultQueueId, input_tensor, memory_config, output_dtype, use_multicore);
    }
};

}  // namespace operations::data_movement

constexpr auto tilize =
    ttnn::register_operation_with_auto_launch_op<"ttnn::tilize", ttnn::operations::data_movement::ExecuteTilize>();

}  // namespace ttnn
