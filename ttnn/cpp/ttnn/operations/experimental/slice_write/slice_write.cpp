// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "slice_write.hpp"
#include "device/slice_write_op.hpp"
#include "tt-metalium/assert.hpp"
#include "tt-metalium/constants.hpp"
#include <tt-logger/tt-logger.hpp>
#include "tt-metalium/math.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor SliceWriteOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& output_tensor,
    const ttnn::SmallVector<uint32_t>& begins,
    const ttnn::SmallVector<uint32_t>& ends,
    const ttnn::SmallVector<uint32_t>& step) {
    const auto& logical_input_shape = input_tensor.logical_shape();
    const auto& padded_input_shape = input_tensor.padded_shape();
    const auto& padded_output_shape = output_tensor.padded_shape();

    bool no_step = std::all_of(step.begin(), step.end(), [](uint32_t s) { return s == 1; });

    TT_FATAL(no_step, "Slice Write does not support strides");

    bool rm_only = !no_step && input_tensor.layout() == Layout::TILE;
    ttnn::Tensor input = input_tensor;
    if (rm_only) {
        input = ttnn::to_layout(input_tensor, Layout::ROW_MAJOR);
    }

    TT_FATAL(!output_tensor.is_sharded(), "Slice Write currently doesn't support sharded output tensors.");
    const bool tiled = input.layout() == Layout::TILE;
    bool on_device = input.storage_type() == StorageType::DEVICE;

    ttnn::SmallVector<uint32_t> padded_ends = ends;
    if (tiled && ends.size() >= 2) {
        // Only pad the last two dimensions for tiled layout
        size_t rank = ends.size();
        padded_ends[rank - 2] =
            std::max(tt::round_up(ends[rank - 2], tt::constants::TILE_HEIGHT), tt::constants::TILE_HEIGHT);
        padded_ends[rank - 1] =
            std::max(tt::round_up(ends[rank - 1], tt::constants::TILE_WIDTH), tt::constants::TILE_WIDTH);
    }
    ttnn::SmallVector<uint32_t> actual_shape_vec;
    ttnn::SmallVector<uint32_t> padded_shape_vec;
    actual_shape_vec.reserve(ends.size());
    padded_shape_vec.reserve(ends.size());

    bool empty = false;
    for (size_t i = 0; i < ends.size(); ++i) {
        TT_FATAL(ends[i] >= begins[i], "End {} must be greater than or equal to start {}", ends[i], begins[i]);
        uint32_t offset = step[i] - begins[i] - 1;
        uint32_t dim_size = (ends[i] + offset) / step[i];
        empty |= dim_size == 0;
        actual_shape_vec.push_back(dim_size);
        padded_shape_vec.push_back(std::max((padded_ends[i] + offset) / step[i], 1u));
    }
    ttnn::Shape actual_shape(actual_shape_vec);
    ttnn::Shape padded_shape(padded_shape_vec);

    if (empty) {
        log_debug(tt::LogOp, "Empty tensor slice, returning unchanged output tensor");
        return output_tensor;
    }

    // Sharding is only 2D.
    if (input.is_sharded() && logical_input_shape[0] == 1 && logical_input_shape[1] == 1) {
        auto shard_spec = input_tensor.shard_spec().value();

        auto input_cores = shard_spec.grid;
        bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
        bool is_block_sharded = input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
        auto total_cores = shard_spec.grid;
        uint32_t num_cores_nhw = total_cores.num_cores();
        if (is_block_sharded) {
            if (rm_orientation) {
                num_cores_nhw = total_cores.bounding_box().grid_size().y;
            } else {
                num_cores_nhw = total_cores.bounding_box().grid_size().x;
            }
        }

    } else {
        for (int i = 0; i < input.logical_shape().rank(); i++) {
            TT_FATAL(
                actual_shape[i] == input.logical_shape()[i],
                "Size of the slice being written {} should match the size of the input tensor {} at dim {}. Got {}, "
                "expected {} , {}",
                actual_shape[i],
                input.logical_shape()[i],
                i,
                actual_shape,
                input.logical_shape(),
                padded_shape);
        }
    }

    if (on_device) {
        const auto& memory_config = output_tensor.memory_config();

        // Check for in-place unpad optimization
        if (input.is_sharded() && input.memory_config() == memory_config && padded_input_shape.rank() > 1) {
            TT_FATAL(no_step, "Sharded tensor slice implementation does not support striding");
            bool in_place_unpad = true;
            for (int i = 0; i < 2; ++i) {
                in_place_unpad &= begins[i] == 0 && ends[i] == 1 && padded_output_shape[i] == 1;
            }
            in_place_unpad &=
                begins[2] == 0 && tt::div_up(ends[2], input.shard_spec().value().shape[0]) ==
                                      tt::div_up(padded_output_shape[2], input.shard_spec().value().shape[0]);
            in_place_unpad &= begins[3] == 0 && ends[3] == padded_output_shape[3];
            if (in_place_unpad) {
                log_info(tt::LogOp, "In-place unpad optimization via copy");
                ttnn::copy(input_tensor, output_tensor);
                return output_tensor;
            }
        }
        log_debug(tt::LogOp, "Invoking SliceWriteDeviceOperation");

        (void)tt::tt_metal::operation::run(
            SliceWriteDeviceOperation{ttnn::Shape(begins), ttnn::Shape(padded_ends), ttnn::Shape(step)},
            {input},
            {},
            {output_tensor})[0];
        return output_tensor;
    }

    TT_THROW("Expects Input on Device");
    return output_tensor;
}

}  // namespace ttnn::operations::experimental
