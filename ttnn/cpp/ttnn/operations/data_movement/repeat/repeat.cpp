// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/data_movement/repeat/repeat.hpp"
#include "device/repeat_op.hpp"

namespace ttnn::operations::data_movement {


ttnn::Tensor RepeatOperation::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const Shape & repeat_dims,
    const std::optional<MemoryConfig>& memory_config_arg) {

    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    operation::launch_op(
        [repeat_dims, memory_config_arg] (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) -> std::vector<Tensor> {
            auto& input_tensor = input_tensors.at(0);
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
                output = operation::run_without_autoformat(RepeatDeviceOperation{dim, repeat_dims[dim], memory_config}, {output}).at(0);
            }
            return {output};
        }, {input_tensor}, output_tensors);
    return output_tensors.at(0);

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
