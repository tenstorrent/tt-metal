// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/data_movement/repeat/repeat.hpp"
#include "device/repeat_op.hpp"
#include "ttnn/operations/data_movement/untilize/untilize.hpp"
#include "ttnn/operations/data_movement/tilize/tilize.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "tt_metal/common/math.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"

namespace ttnn::operations::data_movement {


ttnn::Tensor RepeatOperation::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const Shape & repeat_dims,
    const std::optional<MemoryConfig>& memory_config_arg) {

    auto padded_input_shape = input_tensor.get_padded_shape();
    auto logical_input_shape = input_tensor.get_logical_shape();
    auto input_rank = logical_input_shape.rank();

    auto repeated_logical_shape = logical_input_shape;
    for (uint32_t dim = 0; dim < input_rank; ++dim) {
        repeated_logical_shape[dim] *= repeat_dims[dim];
    }

    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    operation::launch_op(
        [&input_rank,
         &input_tensor,
         &repeat_dims,
         &memory_config_arg,
         &padded_input_shape] (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) -> std::vector<Tensor> {
            auto memory_config = memory_config_arg.value_or(input_tensor.memory_config());
            TT_FATAL(repeat_dims.rank() == input_rank, "Number of repeat dims must be equal to number of tensor dims");
            Tensor output = input_tensor;
            for (uint32_t dim = 0; dim < repeat_dims.size(); ++dim) {
                if (repeat_dims[dim] == 1) {
                    continue;
                }
                TT_FATAL(repeat_dims[dim] > 0, "Number of repetitions along a dim must be greater than 0");
                if (input_tensor.get_layout() == Layout::ROW_MAJOR && dim == input_rank - 1) {
                    TT_FATAL(
                        (padded_input_shape[dim] * input_tensor.element_size()) % input_tensor.buffer()->alignment() == 0,
                        "Current repeat implementation requires aligned last dim when repeating on last dim");
                }
                auto outputs = operation::run_without_autoformat(RepeatDeviceOperation{dim, repeat_dims[dim], memory_config}, {output});
                TT_FATAL(outputs.size() == 1, "ttnn.repeat: expected 1 output tensor from run_without_autoformat, but got {}", outputs.size());
                output = outputs[0];
            }
            return {output};
        }, {}, output_tensors);
    TT_FATAL(output_tensors.size() == 1, "ttnn.repeat: expected 1 output tensor, but got {}", output_tensors.size());
    if (input_tensor.get_layout() != Layout::ROW_MAJOR
        && logical_input_shape != padded_input_shape) {
        auto zero_indices = std::vector<uint32_t>(input_rank, 0);
        auto end_indices = repeated_logical_shape.as_vector();
        auto step = std::vector<uint32_t>(input_rank, 1);

        if (repeated_logical_shape.volume() % tt::constants::TILE_HW != 0) {
            // volume of the repeated tensor doesn't fit neatly into tiles.
            // slice/tilize don't support padding to tiled on the output for
            // now, so we need to perform the slice in row-major then re-tilize
            // ourselves.
            auto rm_output = ttnn::untilize(output_tensors[0]);
            auto sliced_output = ttnn::slice(rm_output, zero_indices, end_indices, step, input_tensor.memory_config(), std::nullopt);

            auto sliced_logical_shape = sliced_output.get_logical_shape();
            auto sliced_padded_shape = sliced_output.get_padded_shape();

            if (sliced_padded_shape.volume() % tt::constants::TILE_HW == 0) {
                // slice preserved tile padding for us, so we can just tilize now.
                auto tiled_output = ttnn::tilize(sliced_output, input_tensor.memory_config());
                return tiled_output;
            }

            auto padded_height = tt::round_up(sliced_padded_shape[-2], tt::constants::TILE_HEIGHT);
            auto padded_width = tt::round_up(sliced_padded_shape[-1], tt::constants::TILE_WIDTH);
            TT_ASSERT(input_rank >= 2, "ttnn.repeat: rank of tiled input tensor must be >= 2");
            uint32_t num_non_hw_dims = input_rank - 2u;
            auto padding_vec = std::vector<std::pair<uint32_t, uint32_t>>(num_non_hw_dims, {0, 0});
            padding_vec.reserve(input_rank);
            padding_vec.emplace_back(0, padded_height - sliced_padded_shape[-2]);
            padding_vec.emplace_back(0, padded_width - sliced_padded_shape[-1]);

            constexpr bool pad_use_multicore = true;
            auto padded_output = ttnn::pad(queue_id, sliced_output, padding_vec, 0.0f, pad_use_multicore, std::nullopt);
            auto tiled_output = ttnn::tilize(padded_output, input_tensor.memory_config());

            auto padded_to_tiled_shape = ttnn::Shape(sliced_logical_shape.as_vector(),
                                                     tiled_output.get_padded_shape().as_vector());
            tiled_output.set_shape(padded_to_tiled_shape);
            return tiled_output;
        } else {
            return ttnn::slice(output_tensors[0], zero_indices, end_indices, step, input_tensor.memory_config(), std::nullopt);
        }
    }
    return output_tensors[0];

}

ttnn::Tensor RepeatOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const Shape & repeat_dims,
    const std::optional<MemoryConfig>& memory_config) {
    return invoke(DefaultQueueId, input_tensor, repeat_dims, memory_config);
}

ttnn::Tensor RepeatOperation::invoke(const ttnn::Tensor& input_tensor, const Shape & repeat_dims) {
    return invoke(DefaultQueueId, input_tensor, repeat_dims, std::nullopt);
}

} // ttnn::operations::data_movement namespace
