// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"
#include "device/sharded_to_interleaved_partial_op.hpp"
#include "sharded_to_interleaved_partial.hpp"

namespace ttnn::operations::data_movement {

ttnn::Tensor ShardedToInterleavedPartialOperation::invoke(uint8_t queue_id,
                                                          const ttnn::Tensor& input_tensor,
                                                          const ttnn::Tensor& cache_tensor,
                                                          int64_t& num_slices,
                                                          int64_t& slice_index,
                                                          const std::optional<MemoryConfig>& memory_config_arg,
                                                          const std::optional<DataType>& data_type_arg) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};

    auto memory_config = memory_config_arg.value_or(input_tensor.memory_config());
    auto shard_spec = input_tensor.shard_spec().value();
    TT_FATAL(input_tensor.shard_spec().has_value(), "Error");
    operation::run(
        ShardedToInterleavedPartialDeviceOperation{.num_slices = num_slices,
                                                   .slice_index = slice_index,
                                                   .output_mem_config = memory_config,
                                                   .output_dtype = data_type_arg.value_or(input_tensor.get_dtype())},
        {input_tensor, cache_tensor});

    return cache_tensor;
}

}  // namespace ttnn::operations::data_movement
