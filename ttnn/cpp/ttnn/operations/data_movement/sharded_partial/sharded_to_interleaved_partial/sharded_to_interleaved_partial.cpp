// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"
#include "device/sharded_to_interleaved_partial_op.hpp"
#include "sharded_to_interleaved_partial.hpp"


namespace ttnn::operations::data_movement{

ttnn::Tensor ShardedToInterleavedPartialOperation::operator()(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& cache_tensor,
    int64_t& num_slices,
    int64_t& slice_index,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DataType> & data_type_arg) {

    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};

    operation::launch_op(
        [num_slices, slice_index, memory_config_arg, data_type_arg] (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
        const auto& input_tensor = input_tensors.at(0);
        auto memory_config = memory_config_arg.value_or(input_tensor.memory_config());
        const auto& output_tensor = input_tensors.at(1);
        auto shard_spec = input_tensor.shard_spec().value();
        TT_FATAL(input_tensor.shard_spec().has_value());
        operation::run(
                ShardedToInterleavedPartialDeviceOperation{
                    .num_slices = num_slices,
                    .slice_index = slice_index,
                    .output_mem_config = memory_config,
                    .output_dtype = data_type_arg.value_or(input_tensor.get_dtype())},
                {input_tensor, output_tensor});
        return {output_tensor};
    }, {input_tensor, cache_tensor}, output_tensors);

    return cache_tensor;



}

} // ttnn::operations::data_movement namespace
