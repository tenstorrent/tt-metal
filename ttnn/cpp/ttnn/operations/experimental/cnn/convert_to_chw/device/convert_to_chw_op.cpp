// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "convert_to_chw_op.hpp"

#include "convert_to_chw_program_factory.hpp"
#include "tt_metal/common/constants.hpp"

namespace ttnn::operations::experimental::cnn {

void ConvertToCHW::validate(const std::vector<Tensor>& input_tensors) const {
    using namespace tt::constants;
    TT_FATAL(input_tensors.size() == 1, "Expected 1 input tensor");

    const auto& input = input_tensors.at(0);
    const auto& shape = input.get_logical_shape();
    const auto& C = shape[-1];
    const auto& HW = shape[-2];

    TT_FATAL(shape.size() == 4, "Input shape must be rank 4 (was rank {})", shape.size());
    TT_FATAL(shape[0] == 1 && shape[1] == 1, "Expected input tensor to be shape [1, 1, HW, C]");
    TT_FATAL(C <= TILE_HEIGHT, "C must be less than or equal to 32 (was {})", C);
    TT_FATAL(HW % TILE_HEIGHT == 0, "HW must be divisible by tile size");

    TT_FATAL(input.is_sharded(), "Input tensor must be sharded");

    const auto& input_shard_spec = input.memory_config().shard_spec.value();
    TT_FATAL(
        input_shard_spec.shape[0] % TILE_HEIGHT == 0,
        "Shard height must be divisible by tile size");  // input shards can be padded so HW may not match shard height
    TT_FATAL(
        this->memory_config.is_sharded() && this->memory_config.memory_layout == TensorMemoryLayout::WIDTH_SHARDED,
        "Output tensor must be width sharded");
}

std::vector<tt::tt_metal::LegacyShape> ConvertToCHW::compute_output_shapes(
    const std::vector<Tensor>& input_tensors) const {
    const auto& shape = input_tensors.at(0).get_logical_shape();
    const auto B = shape[0];
    const auto HW = shape[2];
    const auto C = shape[3];
    return {LegacyShape({B, 1, C, HW}, {B, 1, C, HW})};
}

std::vector<Tensor> ConvertToCHW::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    return operation::generic_create_output_tensors(
        *this, input_tensors, this->dtype, Layout::ROW_MAJOR, this->memory_config);
}

operation::ProgramWithCallbacks ConvertToCHW::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& a = input_tensors.at(0);
    auto& output = output_tensors.at(0);
    auto device_compute_with_storage_grid_size = a.device()->compute_with_storage_grid_size();
    return detail::multi_core_convert_to_chw(a, output, device_compute_with_storage_grid_size);
}
}  // namespace ttnn::operations::experimental::cnn
