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
        // Integer <-> float is a value conversion, not a CB format change. Move first, then typecast in the
        // interleaved destination: casting while still sharded would allocate a second (possibly larger) shard on
        // the same L1 cores and add the sharded typecast's page-size constraints.
        return ttnn::typecast(sharded_to_interleaved(input_tensor, memory_config, std::nullopt), output_dtype.value());
    }

    return ttnn::prim::sharded_to_interleaved(input_tensor, memory_config, output_dtype.value_or(input_tensor.dtype()));
}

}  // namespace ttnn
