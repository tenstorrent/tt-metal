// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "slice_write.hpp"
#include "device/slice_write_op.hpp"
#include "tt-metalium/assert.hpp"
#include "tt-metalium/constants.hpp"
#include <tt-logger/tt-logger.hpp>
#include "tt-metalium/math.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::experimental {

// Specialization for uint32_t and N=4
template <>
ttnn::Tensor SliceWriteOperation::invoke<uint32_t, 4>(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& output_tensor,
    const std::array<uint32_t, 4>& begins,
    const std::array<uint32_t, 4>& ends,
    const std::array<uint32_t, 4>& step) {
    const auto& logical_input_shape = input_tensor.logical_shape();
    const auto& padded_input_shape = input_tensor.padded_shape();
    const auto& padded_output_shape = output_tensor.padded_shape();

    TT_FATAL(padded_input_shape.rank() == 4, "Input tensor must have rank 4");
    TT_FATAL(padded_output_shape.rank() == 4, "Output tensor must have rank 4");

    bool no_step = step[0] == 1 && step[1] == 1 && step[2] == 1 && step[3] == 1;
    bool starts_zero = begins[0] == 0 && begins[1] == 0 && begins[2] == 0 && begins[3] == 0;
    bool ends_max = ends[0] == padded_output_shape[0] && ends[1] == padded_output_shape[1] &&
                    ends[2] == padded_output_shape[2] && ends[3] == padded_output_shape[3];

    TT_FATAL(no_step, "Slice Write does not support strides");

    bool rm_only = !no_step && input_tensor.layout() == Layout::TILE;
    ttnn::Tensor input = input_tensor;
    if (rm_only) {
        input = ttnn::to_layout(input_tensor, Layout::ROW_MAJOR);
    }

    TT_FATAL(!output_tensor.is_sharded(), "Slice Write currently doesn't support sharded output tensors.");
    const bool tiled = input.layout() == Layout::TILE;
    bool on_device = input.storage_type() == StorageType::DEVICE;

    std::array<uint32_t, 4> actual_shape_vec;
    std::array<uint32_t, 4> padded_shape_vec;
    const std::array<uint32_t, 4> padded_ends =
        tiled ? std::array<uint32_t, 4>(
                    {ends[0],
                     ends[1],
                     std::max(tt::round_up(ends[2], tt::constants::TILE_HEIGHT), tt::constants::TILE_HEIGHT),
                     std::max(tt::round_up(ends[3], tt::constants::TILE_WIDTH), tt::constants::TILE_WIDTH)})
              : ends;
    bool empty = false;
    for (int i = 0; i < 4; ++i) {
        TT_FATAL(ends[i] >= begins[i], "End {} must be greater than or equal to start {}", ends[i], begins[i]);
        uint32_t offset = step[i] - begins[i] - 1;
        uint32_t dim_size = (ends[i] + offset) / step[i];
        empty |= dim_size == 0;
        actual_shape_vec[i] = dim_size;
        padded_shape_vec[i] = std::max((padded_ends[i] + offset) / step[i], 1u);
    }
    ttnn::Shape actual_shape(actual_shape_vec);
    ttnn::Shape padded_shape(padded_shape_vec);

    if (empty) {
        log_debug(tt::LogOp, "Empty tensor slice, returning unchanged output tensor");
        return output_tensor;
    }

    // Sharding is only 2D.
    if (input.is_sharded() && logical_input_shape[0] == 1 && logical_input_shape[1] == 1) {
        uint32_t calc_nhw_volume = actual_shape[0] * actual_shape[1] * actual_shape[2];
        auto shard_spec = input_tensor.shard_spec().value();

        auto input_cores = shard_spec.grid;
        auto input_shard_shape = shard_spec.shape;
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

        uint32_t input_nhw_volume = shard_spec.shape[0] * num_cores_nhw;
        uint32_t calc_nhw_volume_padded =
            tt::round_up(tt::div_up(calc_nhw_volume, num_cores_nhw), tt::constants::TILE_HEIGHT) * num_cores_nhw;
    } else {
        for (int i = 0; i < 4; i++) {
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
                ttnn::copy(DefaultQueueId, input_tensor, output_tensor);
                return output_tensor;
            }
        }
        log_debug(tt::LogOp, "Invoking SliceWriteDeviceOperation");

        (void)tt::tt_metal::operation::run(
            SliceWriteDeviceOperation{ttnn::Shape(begins), ttnn::Shape(padded_ends), ttnn::Shape(step)},
            {input},
            {},
            {output_tensor},
            queue_id)[0];
        return output_tensor;
    }

    TT_THROW("Expects Input on Device");
    return output_tensor;
}

}  // namespace ttnn::operations::experimental
