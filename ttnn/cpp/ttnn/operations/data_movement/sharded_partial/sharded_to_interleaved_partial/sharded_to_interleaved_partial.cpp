// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded_partial/sharded_to_interleaved_partial/sharded_to_interleaved_partial.hpp"
#include "ttnn/operations/data_movement/sharded_partial/sharded_to_interleaved_partial/device/sharded_to_interleaved_partial_device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

ttnn::Tensor ShardedToInterleavedPartialOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& cache_tensor,
    int64_t num_slices,
    int64_t slice_index,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DataType>& data_type_arg) {
    auto memory_config = memory_config_arg.value_or(input_tensor.memory_config());
    TT_FATAL(input_tensor.shard_spec().has_value(), "Input tensor must have a shard spec");

    TT_FATAL(num_slices >= 0 && num_slices <= UINT32_MAX, "num_slices must be in range [0, UINT32_MAX]");
    TT_FATAL(slice_index >= 0 && slice_index <= UINT32_MAX, "slice_index must be in range [0, UINT32_MAX]");

    ttnn::prim::sharded_to_interleaved_partial(
        input_tensor,
        cache_tensor,
        static_cast<uint32_t>(num_slices),
        static_cast<uint32_t>(slice_index),
        memory_config,
        data_type_arg.value_or(input_tensor.dtype()));

    return cache_tensor;
}

}  // namespace ttnn::operations::data_movement
