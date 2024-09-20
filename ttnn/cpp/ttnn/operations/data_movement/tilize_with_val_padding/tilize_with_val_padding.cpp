// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_with_val_padding.hpp"

#include "device/tilize_with_val_padding_op.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::data_movement {

ttnn::Tensor ExecuteTilizeWithValPadding::invoke(
    uint8_t queue_id,
    const ttnn::Tensor &input_tensor,
    const tt::tt_metal::LegacyShape &output_tensor_shape,
    float pad_value,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<DataType> output_dtype,
    bool use_multicore) {
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

ttnn::Tensor ExecuteTilizeWithValPadding::invoke(
    const ttnn::Tensor &input_tensor,
    const tt::tt_metal::LegacyShape &output_tensor_shape,
    float pad_value,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<DataType> output_dtype,
    bool use_multicore) {
    return invoke(
        DefaultQueueId, input_tensor, output_tensor_shape, pad_value, memory_config, output_dtype, use_multicore);
}

ttnn::Tensor ExecuteTilizeWithZeroPadding::invoke(
    uint8_t queue_id,
    const ttnn::Tensor &input_tensor,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<DataType> output_dtype,
    bool use_multicore) {
    using namespace tt::constants;
    auto shape = input_tensor.get_legacy_shape();

    shape[2] = tt::round_up(shape[2], TILE_HEIGHT);
    shape[3] = tt::round_up(shape[3], TILE_WIDTH);

    return ExecuteTilizeWithValPadding::invoke(
        queue_id, input_tensor, shape, 0, memory_config, output_dtype, use_multicore);
}

ttnn::Tensor ExecuteTilizeWithZeroPadding::invoke(
    const ttnn::Tensor &input_tensor,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<DataType> output_dtype,
    bool use_multicore) {
    return invoke(DefaultQueueId, input_tensor, memory_config, output_dtype, use_multicore);
}

}  // namespace ttnn::operations::data_movement
