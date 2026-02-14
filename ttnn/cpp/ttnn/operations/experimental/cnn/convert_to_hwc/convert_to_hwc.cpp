// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "convert_to_hwc.hpp"
#include "device/convert_to_hwc_device_operation.hpp"

namespace ttnn::operations::experimental::cnn {

static tt::tt_metal::MemoryConfig infer_hwc_output_memory_config(const ttnn::Tensor& input_tensor) {
    TT_FATAL(input_tensor.is_sharded(), "Input tensor must be sharded to infer output memory config");

    const auto& input_memory_config = input_tensor.memory_config();
    TT_FATAL(
        input_memory_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
        "Input tensor must be width sharded");

    const auto& logical_shape = input_tensor.logical_shape();
    const int B = logical_shape[1];
    const int C = logical_shape[2];

    const auto& input_shard_spec = input_memory_config.shard_spec().value();
    const int input_shard_width = input_shard_spec.shape[1];

    // Per-core output height is B times the per-core HW width (i.e., sticks)
    const int output_shard_height = B * input_shard_width;
    const int alignment_elements = ttnn::experimental::prim::compute_alignment_requirement_in_elements(input_tensor);
    TT_FATAL(alignment_elements != 0, "Number of alignment elements cannot be 0");
    const int output_shard_width = tt::round_up(C, alignment_elements);

    const std::array<uint32_t, 2> output_shard_shape = {output_shard_height, output_shard_width};

    auto output_shard_spec =
        tt::tt_metal::ShardSpec(input_shard_spec.grid, output_shard_shape, input_shard_spec.orientation);

    return tt::tt_metal::MemoryConfig(
        tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        input_tensor.memory_config().buffer_type(),
        output_shard_spec);
}

ttnn::Tensor ExecuteConvertToHWC::invoke(
    const Tensor& a, const std::optional<MemoryConfig>& memory_config, const std::optional<DataType>& dtype) {
    const auto& input_memory_config = a.memory_config();
    const bool is_dram_input = input_memory_config.buffer_type() == tt::tt_metal::BufferType::DRAM;

    tt::tt_metal::MemoryConfig output_memory_config;
    if (is_dram_input) {
        TT_FATAL(
            memory_config.has_value(),
            "When input tensor is in DRAM, output memory_config must be explicitly specified");
        output_memory_config = memory_config.value();
    } else {
        if (memory_config.has_value()) {
            output_memory_config = memory_config.value();
        } else {
            output_memory_config = infer_hwc_output_memory_config(a);
        }
    }
    const auto alignment_elements = ttnn::experimental::prim::compute_alignment_requirement_in_elements(a);
    TT_FATAL(alignment_elements != 0, "Number of alignment elements cannot be 0");
    TT_FATAL(
        output_memory_config.shard_spec()->shape[1] % alignment_elements == 0,
        "Output shard width must be rounded up to next multiple of {} to satisfy alignment constraints (width was {})",
        alignment_elements,
        output_memory_config.shard_spec()->shape[1]);

    return ttnn::prim::convert_to_hwc(a, output_memory_config, dtype.value_or(a.dtype()));
}

}  // namespace ttnn::operations::experimental::cnn
