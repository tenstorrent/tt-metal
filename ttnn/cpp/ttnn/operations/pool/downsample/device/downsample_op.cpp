// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "downsample_op.hpp"
#include "downsample_program_factory.hpp"

#include <math.h>

#include "ttnn/operations/math.hpp"
#include <tt-metalium/constants.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::downsample {

void Downsample::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to downsample need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to downsample need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.get_layout() == Layout::TILE, "Can only downsample tile major data");

    TT_FATAL(input_tensor_a.volume() % TILE_HW == 0, "Error");
    TT_FATAL(input_tensor_a.memory_config().is_sharded(), "Error");
    TT_FATAL(
        input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED ||
            input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED,
        "Unsupported memory layout {}.",
        input_tensor_a.memory_config().memory_layout());
}

std::vector<TensorSpec> Downsample::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_ASSERT(input_tensor.get_padded_shape()[0] == 1 && input_tensor.get_padded_shape()[1] == 1);
    uint32_t input_height = input_tensor.get_padded_shape()[2];
    auto [img_batch_size, img_height, img_width, img_stride_h, img_stride_w] = this->downsample_params;
    TT_ASSERT(input_height >= img_batch_size * img_height * img_width);
    uint32_t output_height = img_batch_size * ceil((double)img_height / (double)img_stride_h) *
                             ceil((double)img_width / (double)img_stride_w);
    uint32_t output_width = input_tensor.get_padded_shape()[3];
    auto output_shape = Shape({1, 1, output_height, output_width});

    auto [num_cores_height_sliced, num_cores_width_sliced] = detail::get_num_cores_height_width_sliced(
        input_tensor.shard_spec().value().grid,
        input_tensor.memory_config().memory_layout(),
        input_tensor.shard_spec().value().orientation);
    uint32_t output_shard_height =
        tt::round_up(output_shape[2], num_cores_height_sliced * TILE_HEIGHT) / num_cores_height_sliced;
    uint32_t output_shard_width =
        tt::round_up(output_shape[3], num_cores_width_sliced * TILE_WIDTH) / num_cores_width_sliced;
    auto mem_config = input_tensor.memory_config().with_shard_spec(ShardSpec{
        input_tensor.shard_spec().value().grid,
        std::array<uint32_t, 2>{{output_shard_height, output_shard_width}},
        input_tensor.shard_spec().value().orientation});
    return {TensorSpec(output_shape, TensorLayout(dtype, PageConfig(Layout::TILE), mem_config))};
}

operation::ProgramWithCallbacks Downsample::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return {detail::downsample_single_core(input_tensor_a, downsample_params, output_tensor)};
}

Tensor downsample(
    const ttnn::Tensor& input_tensor_a, std::array<uint32_t, 5> downsample_params, std::optional<DataType> dtype) {
    auto dtype_ = dtype.has_value() ? dtype.value() : input_tensor_a.get_dtype();
    auto output_tensors = operation::run(Downsample{downsample_params, dtype_}, {input_tensor_a});
    return output_tensors.at(0);
}

}  // namespace ttnn::operations::downsample
