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

namespace ttnn::operations::data_movement {


ttnn::Tensor RepeatOperation::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const Shape & repeat_dims,
    const std::optional<MemoryConfig>& memory_config_arg) {

    // special path for handling padded tiled tensors that are naively repeated
    // *with padding* in the current implementation
    std::optional<Tensor> formatted_input_tensor = std::nullopt;
    if (input_tensor.get_layout() != Layout::ROW_MAJOR
        && input_tensor.get_legacy_shape().without_padding() != input_tensor.get_legacy_shape()) {
        formatted_input_tensor = ttnn::untilize(input_tensor);
    }

    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({formatted_input_tensor.has_value() ? *formatted_input_tensor : input_tensor}))};
    operation::launch_op(
        [repeat_dims, memory_config_arg] (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) -> std::vector<Tensor> {
            auto& input_tensor = input_tensors[0];
            auto memory_config = memory_config_arg.value_or(input_tensor.memory_config());
            uint32_t input_rank = input_tensor.get_legacy_shape().rank();
            TT_FATAL(repeat_dims.rank() == input_rank, "Number of repeat dims must be equal to number of tensor dims");
            Tensor output = input_tensor;
            for (uint32_t dim = 0; dim < repeat_dims.size(); ++dim) {
                if (repeat_dims[dim] == 1) {
                    continue;
                }
                TT_FATAL(repeat_dims[dim] > 0, "Number of repetitions along a dim must be greater than 0");
                if (input_tensor.get_layout() == Layout::ROW_MAJOR && dim == input_rank - 1) {
                    TT_FATAL(
                        (input_tensor.get_legacy_shape()[dim] * input_tensor.element_size()) % input_tensor.buffer()->alignment() == 0,
                        "Current repeat implementation requires aligned last dim when repeating on last dim");
                }
                auto outputs = operation::run_without_autoformat(RepeatDeviceOperation{dim, repeat_dims[dim], memory_config}, {output});
                TT_FATAL(outputs.size() == 1, "ttnn.repeat: expected 1 output tensor from run_without_autoformat, but got {}", outputs.size());
                output = outputs[0];
            }
            return {output};
        }, {formatted_input_tensor.has_value() ? *formatted_input_tensor : input_tensor}, output_tensors);
    TT_FATAL(output_tensors.size() == 1, "ttnn.repeat: expected 1 output tensor, but got {}", output_tensors.size());
    if (formatted_input_tensor.has_value()) {
        auto zero_indices = std::vector<uint32_t>(input_tensor.get_legacy_shape().rank(), 0);
        std::vector<uint32_t> end_indices(repeat_dims.size());
        for (uint32_t i = 0; i < repeat_dims.size(); ++i) {
            end_indices[i] = repeat_dims[i] * input_tensor.get_legacy_shape().without_padding()[i];
        }
        auto step = std::vector<uint32_t>(input_tensor.get_legacy_shape().rank(), 1);
        auto sliced_output = ttnn::slice(output_tensors[0], zero_indices, end_indices, step, input_tensor.memory_config(), std::nullopt);
        auto padded_to_tile = sliced_output.cpu().pad_to_tile(0.0).to(input_tensor.device(), input_tensor.memory_config());
        auto tiled_output = ttnn::tilize(padded_to_tile);
        return tiled_output;
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
