// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "slice_write.hpp"
#include "device/slice_write_op.hpp"
#include "tt-metalium/assert.hpp"
#include "tt-metalium/logger.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "cpp/ttnn/operations/creation.hpp"
#include "cpp/ttnn/operations/data_movement/copy/copy.hpp"
#include "cpp/ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "cpp/ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::data_movement {

// Specialization for uint32_t and N=4
template <>
ttnn::Tensor SliceWriteOperation::invoke<uint32_t, 4>(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& output_tensor,
    const std::array<uint32_t, 4>& begins,
    const std::array<uint32_t, 4>& ends,
    const std::array<uint32_t, 4>& step) {
    const auto& padded_input_shape = input_tensor.get_padded_shape();
    const auto& padded_output_shape = output_tensor.get_padded_shape();

    TT_FATAL(padded_input_shape.rank() == 4, "Input tensor must have rank 4");
    TT_FATAL(padded_output_shape.rank() == 4, "Output tensor must have rank 4");

    bool no_step = step[0] == 1 && step[1] == 1 && step[2] == 1 && step[3] == 1;
    bool starts_zero = begins[0] == 0 && begins[1] == 0 && begins[2] == 0 && begins[3] == 0;
    bool ends_max = ends[0] == padded_output_shape[0] && ends[1] == padded_output_shape[1] &&
                    ends[2] == padded_output_shape[2] && ends[3] == padded_output_shape[3];

    // if (no_step && starts_zero && ends_max) {
    //     ttnn::copy(queue_id, input_tensor, output_tensor);
    //     return output_tensor;
    // }
    bool rm_only = !no_step && input_tensor.get_layout() == Layout::TILE;
    ttnn::Tensor input = input_tensor;
    if (rm_only) {
        input = ttnn::to_layout(input_tensor, Layout::ROW_MAJOR, std::nullopt, std::nullopt, (IDevice*)nullptr);
    }
    TT_FATAL(
        (!input_tensor.is_sharded()) || (input_tensor.is_sharded() && input_tensor.memory_config().memory_layout ==
                                                                          TensorMemoryLayout::HEIGHT_SHARDED),
        "Slice Write currently supports only Height Sharding for input tensors.");

    TT_FATAL(!output_tensor.is_sharded(), "Slice Write currently doesn't support sharded output tensors.");
    const bool tiled = input.get_layout() == Layout::TILE;
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
        tt::log_debug("Empty tensor slice, returning unchanged output tensor");
        return output_tensor;
    }

    // // Early exit if slice is a no-op
    // if (padded_shape == padded_output_shape && no_step) {
    //     ttnn::copy(queue_id, input_tensor, output_tensor);
    //     tt::log_debug("Input Tensor same shape as output tensor. Performing copy");
    //     return output_tensor;
    // }

    for (int i = 0; i < 4; i++) {
        TT_FATAL(
            actual_shape[i] == input.get_logical_shape()[i],
            "Size of the slice being written {} should match the size of the input tensor {} at dim {}. Got {}, "
            "expected {} , {}",
            actual_shape[i],
            input.get_logical_shape()[i],
            i,
            actual_shape,
            input.get_logical_shape(),
            padded_shape);
    }

    if (on_device) {
        auto memory_config = output_tensor.memory_config();

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
                tt::log_info("In-place unpad optimization via copy");
                ttnn::copy(DefaultQueueId, input_tensor, output_tensor);
                return output_tensor;
            }
        }
        tt::log_debug("Invoking SliceWriteDeviceOperation");

        tt::tt_metal::operation::run(
            SliceWriteDeviceOperation{ttnn::Shape(begins), ttnn::Shape(padded_ends), ttnn::Shape(step)},
            {input},
            {},
            {output_tensor},
            queue_id)[0];
        return output_tensor;
    }

    TT_FATAL(no_step, "Host tensor slice does not support strides");
    TT_THROW("Expects Input on Device");
    return output_tensor;
}

template <typename T, std::size_t N>
ttnn::Tensor SliceWriteOperation::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& output_tensor,
    const std::array<T, N>& output_tensor_start,
    const std::array<T, N>& output_tensor_end,
    const std::array<T, N>& step) {
    tt::stl::Span<const T> start(output_tensor_start.begin(), output_tensor_start.end());
    tt::stl::Span<const T> end(output_tensor_end.begin(), output_tensor_end.end());
    tt::stl::Span<const T> step_vec(step.begin(), step.end());
    return SliceWriteOperation::invoke<T>(queue_id, input_tensor, output_tensor, start, end, step_vec);
}

template <typename T, std::size_t N>
ttnn::Tensor SliceWriteOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& output_tensor,
    const std::array<T, N>& output_tensor_start,
    const std::array<T, N>& output_tensor_end,
    const std::array<T, N>& step) {
    return SliceWriteOperation::invoke<T, N>(
        ttnn::DefaultQueueId, input_tensor, output_tensor, output_tensor_start, output_tensor_end, step);
}

template ttnn::Tensor SliceWriteOperation::invoke<uint32_t, 4>(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& output_tensor,
    const std::array<uint32_t, 4>& output_tensor_start,
    const std::array<uint32_t, 4>& output_tensor_end,
    const std::array<uint32_t, 4>& step);

}  // namespace ttnn::operations::data_movement
