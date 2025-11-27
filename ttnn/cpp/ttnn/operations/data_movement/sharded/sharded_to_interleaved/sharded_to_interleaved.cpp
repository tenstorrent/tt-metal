// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sharded_to_interleaved.hpp"
#include "device/sharded_to_interleaved_device_operation.hpp"

namespace ttnn::operations::data_movement {

ttnn::Tensor ShardedToInterleavedOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const MemoryConfig& memory_config,
    const std::optional<DataType>& output_dtype,
    const std::optional<bool>& is_l1_aligned) {
    if (!input_tensor.shard_spec().has_value()) {
        return input_tensor;
    }

    return ttnn::prim::sharded_to_interleaved(
        input_tensor, memory_config, output_dtype.value_or(input_tensor.dtype()), is_l1_aligned.value_or(false));
}

}  // namespace ttnn::operations::data_movement
