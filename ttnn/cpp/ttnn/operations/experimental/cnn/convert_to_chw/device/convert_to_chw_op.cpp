// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "convert_to_chw_op.hpp"

#include "convert_to_chw_program_factory.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::experimental::cnn {

tt::tt_metal::MemoryConfig infer_output_memory_config(const Tensor& input_tensor) {
    using namespace tt::constants;

    TT_FATAL(input_tensor.is_sharded(), "Input tensor must be sharded to infer output memory config");

    const auto& input_shard_spec = input_tensor.memory_config().shard_spec().value();
    const auto& input_shape = input_tensor.logical_shape();
    const auto C = input_shape[-1];

    // Calculate output shard dimensions
    // Input tensor: [1, 1, HW, C] with HEIGHT_SHARDED [shard_height, 32]
    // Output tensor: [1, 1, C, HW] with WIDTH_SHARDED [C, shard_width_per_core]
    //
    // For HEIGHT_SHARDED input, HW is distributed across cores:
    //   HW_per_core = shard_height (since we're height sharding the HW dimension)
    // For WIDTH_SHARDED output, HW should still be distributed the same way:
    //   output_shard_width = HW_per_core = shard_height
    const auto input_shard_height = input_shard_spec.shape[0];
    const auto output_shard_width = input_shard_height;  // HW dimension per core stays the same

    // Create output shard spec with WIDTH_SHARDED layout
    const std::array<uint32_t, 2> output_shard_shape = {C, output_shard_width};
    auto output_shard_spec =
        tt::tt_metal::ShardSpec(input_shard_spec.grid, output_shard_shape, input_shard_spec.orientation);

    return tt::tt_metal::MemoryConfig(
        tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED, input_tensor.memory_config().buffer_type(), output_shard_spec);
}

void ConvertToCHW::validate(const std::vector<Tensor>& input_tensors) const {
    using namespace tt::constants;
    TT_FATAL(input_tensors.size() == 1, "Expected 1 input tensor");

    const auto& input = input_tensors.at(0);
    const auto& shape = input.logical_shape();
    const auto& C = shape[-1];
    const auto& HW = shape[-2];

    TT_FATAL(shape.size() == 4, "Input shape must be rank 4 (was rank {})", shape.size());
    TT_FATAL(shape[0] == 1 && shape[1] == 1, "Expected input tensor to be shape [1, 1, HW, C]");
    TT_FATAL(C <= TILE_HEIGHT, "C must be less than or equal to 32 (was {})", C);
    TT_FATAL(HW % TILE_HEIGHT == 0, "HW must be divisible by tile size");

    TT_FATAL(input.is_sharded(), "Input tensor must be sharded");

    const auto& input_shard_spec = input.memory_config().shard_spec().value();
    TT_FATAL(
        input_shard_spec.shape[0] % TILE_HEIGHT == 0,
        "Shard height must be divisible by tile size");  // input shards can be padded so HW may not match shard height
    TT_FATAL(
        this->memory_config.is_sharded() &&
            this->memory_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
        "Output tensor must be width sharded");
}

std::vector<ttnn::TensorSpec> ConvertToCHW::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& shape = input_tensors.at(0).logical_shape();
    const auto B = shape[0];
    const auto HW = shape[2];
    const auto C = shape[3];
    return {TensorSpec(
        Shape({B, 1, C, HW}),
        tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR), memory_config))};
}

tt::tt_metal::operation::ProgramWithCallbacks ConvertToCHW::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& a = input_tensors.at(0);
    auto& output = output_tensors.at(0);
    auto device_compute_with_storage_grid_size = a.device()->compute_with_storage_grid_size();
    return detail::multi_core_convert_to_chw(a, output, device_compute_with_storage_grid_size);
}
}  // namespace ttnn::operations::experimental::cnn
