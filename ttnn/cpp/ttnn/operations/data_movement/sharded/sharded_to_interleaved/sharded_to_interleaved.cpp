// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved.hpp"
#include "device/sharded_to_interleaved_device_operation.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"

namespace ttnn {

ttnn::Tensor sharded_to_interleaved(
    const ttnn::Tensor& input_tensor, const MemoryConfig& memory_config, const std::optional<DataType>& output_dtype) {
    if (!input_tensor.shard_spec().has_value()) {
        return input_tensor;
    }

    if (output_dtype.has_value() && is_int_float_conversion(input_tensor.dtype(), output_dtype.value())) {
        // Integer <-> float is a value conversion, not a CB format change: typecast first, then move.
        return sharded_to_interleaved(ttnn::typecast(input_tensor, output_dtype.value()), memory_config, std::nullopt);
    }

    return ttnn::prim::sharded_to_interleaved(input_tensor, memory_config, output_dtype.value_or(input_tensor.dtype()));
}

}  // namespace ttnn
