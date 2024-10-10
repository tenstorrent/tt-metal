// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/data_movement/repeat/repeat.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/interleaved_to_sharded.hpp"
#include "device/repeat_op.hpp"

namespace ttnn::operations::data_movement {


ttnn::Tensor RepeatOperation::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const Shape & repeat_dims,
    const std::optional<MemoryConfig>& memory_config_arg) {

    ttnn::Tensor interleaved_input_tensor;


    // Currently no native support for sharded tensors
    // Converting to Interleaved before sending to device operation
    if(input_tensor.is_sharded()) {
        tt::log_warning(tt::LogOp, "Sharding is not natively supported, converting to interleaved before going to device, and then will shard output");
        auto dram_memory_config = tt::tt_metal::MemoryConfig{.memory_layout=tt::tt_metal::TensorMemoryLayout::INTERLEAVED,.buffer_type=tt::tt_metal::BufferType::DRAM};
        interleaved_input_tensor = ttnn::sharded_to_interleaved(queue_id, input_tensor, dram_memory_config, std::nullopt);
    }
    else {
        interleaved_input_tensor = input_tensor;
    }

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
        }, {interleaved_input_tensor}, output_tensors);

    // If input was originally sharded, we want to shard the output again.
    if(input_tensor.is_sharded()) {
        auto shard_spec = input_tensor.shard_spec().value();
        for(uint32_t dim_idx = 0; dim_idx < repeat_dims.rank() - 1; dim_idx++) {
            shard_spec.shape[0] *= repeat_dims[dim_idx];
        }
        shard_spec.shape[1] *= repeat_dims[repeat_dims.rank() - 1];
        return ttnn::interleaved_to_sharded(queue_id, output_tensors.at(0), shard_spec.grid, shard_spec.shape, input_tensor.buffer()->buffer_layout(), shard_spec.orientation, std::nullopt);
    }
    else {
        return output_tensors.at(0);
    }

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
