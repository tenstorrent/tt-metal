// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"
#include "device/sharded_to_interleaved_op.hpp"
#include "sharded_to_interleaved.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

ttnn::Tensor ShardedToInterleavedOperation::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const MemoryConfig& memory_config,
    const std::optional<DataType>& output_dtype,
    const std::optional<bool>& is_l1_aligned) {
    if (!input_tensor.shard_spec().has_value()) {
        return input_tensor;
    }

    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    auto shard_spec = input_tensor.shard_spec().value();
    return operation::run(
               ShardedToInterleavedDeviceOperation{
                   .output_mem_config = memory_config,
                   .output_dtype = output_dtype.value_or(input_tensor.get_dtype()),
                   .is_l1_aligned = is_l1_aligned.value_or(false)},
               {input_tensor})
        .at(0);
}

}  // namespace ttnn::operations::data_movement
