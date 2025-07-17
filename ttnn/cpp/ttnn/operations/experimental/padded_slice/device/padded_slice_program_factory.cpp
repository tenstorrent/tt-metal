// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "padded_slice_program_factory.hpp"

namespace ttnn::operations::experimental::detail {

uint32_t get_num_cores_channels_from_sharded_tensor(const Tensor& tensor) {
    auto shard_spec = tensor.shard_spec().value();
    auto core_grid = shard_spec.grid;

    bool rm_orientation = shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR;

    uint32_t num_cores_channels = 1;
    if (tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED) {
        if (rm_orientation) {
            num_cores_channels = core_grid.bounding_box().grid_size().x;
        } else {
            num_cores_channels = core_grid.bounding_box().grid_size().y;
        }
    } else if (tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED) {
        num_cores_channels = core_grid.num_cores();
    }
    return num_cores_channels;
}

tt::tt_metal::operation::ProgramWithCallbacks padded_slice_multi_core(
    const Tensor& a,
    Tensor& output,
    const ttnn::Shape& output_tensor_start,
    const ttnn::Shape& output_tensor_end,
    const ttnn::Shape& step) {
    bool has_step = false;
    for (int i = 0; i < step.size(); i++) {
        if (step[i] != 1) {
            has_step = true;
            break;
        }
    }
    TT_FATAL(!has_step, "Padded Slice with Stride is not supported yet");
    TT_FATAL(
        output.is_sharded(),
        "Output must be sharded for the padded_slice operation. Use slice for non-sharded outputs");
    TT_FATAL(!a.is_sharded(), " Sharded input is not supported for padded_slice operation");
    if (a.layout() == tt::tt_metal::Layout::ROW_MAJOR) {
        return padded_slice_rm_multi_core(a, output, output_tensor_start, output_tensor_end);
    } else if (a.layout() == tt::tt_metal::Layout::TILE) {
        return padded_slice_tile_multi_core(a, output, output_tensor_start, output_tensor_end);
    } else {
        TT_THROW("Unsupported layout for padded_slice operation: {}", a.layout());
    }
    return {};
}

}  // namespace ttnn::operations::experimental::detail
