// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"
#include "device/sharded_to_interleaved_op.hpp"
#include "sharded_to_interleaved.hpp"

namespace ttnn::operations::data_movement{

ttnn::Tensor ShardedToInterleavedOperation::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const MemoryConfig & memory_config,
    const std::optional<DataType> & output_dtype
    ) {

    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};

    auto shard_spec = input_tensor.shard_spec().value();
    TT_FATAL(input_tensor.shard_spec().has_value(), "Error");
    return operation::run(
            ShardedToInterleavedDeviceOperation{
                .output_mem_config = memory_config,
                .output_dtype = output_dtype.value_or(input_tensor.get_dtype())
                },
            {input_tensor}).at(0);

}

} // ttnn::operations::data_movement namespace
