// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved.hpp"
#include "device/sharded_to_interleaved_device_operation.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

using ttnn::operations::data_movement::uint16_output_dtype_needs_typecast;

namespace ttnn {

ttnn::Tensor sharded_to_interleaved(
    const ttnn::Tensor& input_tensor, const MemoryConfig& memory_config, const std::optional<DataType>& output_dtype) {
    if (!input_tensor.shard_spec().has_value()) {
        return input_tensor;
    }

    const DataType resolved_dtype = output_dtype.value_or(input_tensor.dtype());
    // Move at the source dtype (an exact copy) and convert by value on the interleaved result, which
    // is the input typecast is tested against: converting first would leave the shard spec to be
    // re-derived for a dtype the caller never asked to shard.
    if (uint16_output_dtype_needs_typecast(input_tensor.dtype(), resolved_dtype)) {
        ttnn::Tensor moved = ttnn::prim::sharded_to_interleaved(input_tensor, memory_config, input_tensor.dtype());
        return ttnn::typecast(moved, resolved_dtype, memory_config);
    }

    return ttnn::prim::sharded_to_interleaved(input_tensor, memory_config, resolved_dtype);
}

}  // namespace ttnn
