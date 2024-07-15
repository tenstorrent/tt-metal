// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/tilize_with_val_padding_op.hpp"
#include "tt_eager/tt_dnn/op_library/run_operation.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::data_movement {

struct ExecuteTilizeWithValPadding {
    static ttnn::Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const ttnn::Tensor &input_tensor,
        const tt::tt_metal::Shape &output_tensor_shape,
        float pad_value,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<DataType> output_dtype = std::nullopt,
        bool use_multicore = false) {
        return operation::run(
                   TilizeWithValPadding{
                       output_tensor_shape,
                       pad_value,
                       memory_config.value_or(input_tensor.memory_config()),
                       output_dtype.value_or(input_tensor.get_dtype()),
                       use_multicore},
                   {input_tensor},
                   {},
                   {},
                   queue_id)
            .at(0);
    }

    static ttnn::Tensor execute_on_worker_thread(
        const ttnn::Tensor &input_tensor,
        const tt::tt_metal::Shape &output_tensor_shape,
        float pad_value,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<DataType> output_dtype = std::nullopt,
        bool use_multicore = false) {
        constexpr uint8_t DefaultQueueId = 0;
        return execute_on_worker_thread(
            DefaultQueueId, input_tensor, output_tensor_shape, pad_value, memory_config, output_dtype, use_multicore);
    }
};

struct ExecuteTilizeWithZeroPadding {
    static constexpr uint8_t DefaultQueueId = 0;

    static ttnn::Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const ttnn::Tensor &input_tensor,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<DataType> output_dtype = std::nullopt,
        bool use_multicore = false) {
        auto shape = input_tensor.get_legacy_shape();

        shape[2] = tt::round_up(shape[2], TILE_HEIGHT);
        shape[3] = tt::round_up(shape[3], TILE_WIDTH);

        return ExecuteTilizeWithValPadding::execute_on_worker_thread(
            queue_id, input_tensor, shape, 0, memory_config, output_dtype, use_multicore);
    }

    static ttnn::Tensor execute_on_worker_thread(
        const ttnn::Tensor &input_tensor,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<DataType> output_dtype = std::nullopt,
        bool use_multicore = false) {
        constexpr uint8_t DefaultQueueId = 0;
        return execute_on_worker_thread(DefaultQueueId, input_tensor, memory_config, output_dtype, use_multicore);
    }
};

}  // namespace operations::data_movement

constexpr auto tilize_with_val_padding =
    ttnn::register_operation<ttnn::operations::data_movement::ExecuteTilizeWithValPadding>(
        "ttnn::tilize_with_val_padding");

constexpr auto tilize_with_zero_padding =
    ttnn::register_operation<ttnn::operations::data_movement::ExecuteTilizeWithZeroPadding>(
        "ttnn::tilize_with_zero_padding");

}  // namespace ttnn
