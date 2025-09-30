// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "convert_to_chw.hpp"

#include "device/convert_to_chw_op.hpp"

namespace ttnn::operations::experimental::cnn {

static tt::tt_metal::MemoryConfig infer_output_memory_config(const Tensor& input_tensor) {
    using namespace tt::constants;

    TT_FATAL(input_tensor.is_sharded(), "Input tensor must be sharded to infer output memory config");

    const auto& input_memory_config = input_tensor.memory_config();
    TT_FATAL(
        input_memory_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        "Input tensor must be height sharded");

    const auto& input_shape = input_tensor.logical_shape();
    const auto C = input_shape[-1];

    const auto& input_shard_spec = input_memory_config.shard_spec().value();
    const auto input_shard_height = input_shard_spec.shape[0];
    const auto output_shard_width = input_shard_height;  // HW dimension per core stays the same

    const std::array<uint32_t, 2> output_shard_shape = {C, output_shard_width};
    auto output_shard_spec =
        tt::tt_metal::ShardSpec(input_shard_spec.grid, output_shard_shape, input_shard_spec.orientation);

    return tt::tt_metal::MemoryConfig(
        tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED, input_tensor.memory_config().buffer_type(), output_shard_spec);
}

ttnn::Tensor ExecuteConvertToCHW::invoke(const Tensor& a, const std::optional<DataType>& dtype) {
    const auto output_memory_config = infer_output_memory_config(a);
    auto program = ConvertToCHW{output_memory_config, dtype.value_or(a.dtype())};
    return tt::tt_metal::operation::run(program, {a}, {}, {}).at(0);
}

}  // namespace ttnn::operations::experimental::cnn
