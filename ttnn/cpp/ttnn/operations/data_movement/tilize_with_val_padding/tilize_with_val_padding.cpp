// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_with_val_padding.hpp"

#include "device/tilize_with_val_padding_op.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

ttnn::Tensor ExecuteTilizeWithValPadding::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::SimpleShape& output_tensor_shape,
    const PadValue pad_value,
    const std::optional<MemoryConfig>& memory_config,
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
    const ttnn::Tensor& input_tensor,
    const ttnn::SimpleShape& output_tensor_shape,
    const PadValue pad_value,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype,
    bool use_multicore) {
    return invoke(
        DefaultQueueId, input_tensor, output_tensor_shape, pad_value, memory_config, output_dtype, use_multicore);
}

ttnn::Tensor ExecuteTilizeWithValPadding::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::SmallVector<uint32_t>& output_tensor_shape,
    const PadValue pad_value,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype,
    bool use_multicore) {
    return operation::run(
               TilizeWithValPadding{
                   ttnn::SimpleShape(output_tensor_shape),
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
    const ttnn::Tensor& input_tensor,
    const ttnn::SmallVector<uint32_t>& output_tensor_shape,
    const PadValue pad_value,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype,
    bool use_multicore) {
    return invoke(
        DefaultQueueId, input_tensor, output_tensor_shape, pad_value, memory_config, output_dtype, use_multicore);
}

ttnn::Tensor ExecuteTilizeWithValPadding::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const PadValue pad_value,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype,
    bool use_multicore) {
    using namespace tt::constants;
    std::array<uint32_t, 4> shape;
    auto input_shape = input_tensor.get_shape();
    shape[0] = input_shape[0];
    shape[1] = input_shape[1];
    shape[2] = tt::round_up(shape[2], TILE_HEIGHT);
    shape[3] = tt::round_up(shape[3], TILE_WIDTH);

    return invoke(
        queue_id, input_tensor, ttnn::SimpleShape(shape), pad_value, memory_config, output_dtype, use_multicore);
}

ttnn::Tensor ExecuteTilizeWithValPadding::invoke(
    const ttnn::Tensor& input_tensor,
    const PadValue pad_value,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype,
    bool use_multicore) {
    return invoke(DefaultQueueId, input_tensor, pad_value, memory_config, output_dtype, use_multicore);
}

ttnn::Tensor ExecuteTilizeWithZeroPadding::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype,
    bool use_multicore) {
    using namespace tt::constants;
    std::array<uint32_t, 4> shape;
    auto input_shape = input_tensor.get_shape();
    shape[0] = input_shape[0];
    shape[1] = input_shape[1];
    shape[2] = tt::round_up(shape[2], TILE_HEIGHT);
    shape[3] = tt::round_up(shape[3], TILE_WIDTH);

    PadValue pad_value;
    if (input_tensor.get_dtype() == DataType::BFLOAT16 or input_tensor.get_dtype() == DataType::FLOAT32) {
        pad_value = 0.0f;
    } else {
        pad_value = (uint32_t)0;
    }
    return ExecuteTilizeWithValPadding::invoke(
        queue_id, input_tensor, ttnn::SimpleShape(shape), pad_value, memory_config, output_dtype, use_multicore);
}

ttnn::Tensor ExecuteTilizeWithZeroPadding::invoke(
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype,
    bool use_multicore) {
    return invoke(DefaultQueueId, input_tensor, memory_config, output_dtype, use_multicore);
}

}  // namespace ttnn::operations::data_movement
