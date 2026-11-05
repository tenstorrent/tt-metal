// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"  // ttnn::typecast zero-extends narrow integer sources
#include "ttnn/operations/data_movement/copy/device/copy_device_operation.hpp"

namespace ttnn::experimental {

namespace {
bool copy_cannot_zero_extend(const DataType& input_dtype, const DataType& output_dtype) {
    return (input_dtype == DataType::UINT16 or input_dtype == DataType::UINT8) and
           (output_dtype == DataType::UINT32 or output_dtype == DataType::INT32);
}
}  // namespace

ttnn::Tensor typecast(
    const Tensor& input_tensor,
    const DataType& dtype,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& optional_output_tensor) {
    // ttnn::prim::copy's copy_tile path does not zero-extend narrow integer sources, so widening a
    // uint16/uint8 tensor to a 32-bit integer output returns shifted values. Route those through
    // ttnn::typecast, which zero-extends correctly; every other conversion keeps the copy path.
    if (copy_cannot_zero_extend(input_tensor.dtype(), dtype)) {
        return ttnn::typecast(input_tensor, dtype, output_mem_config, optional_output_tensor);
    }
    return ttnn::prim::copy(
        input_tensor, output_mem_config.value_or(input_tensor.memory_config()), dtype, optional_output_tensor);
}

}  // namespace ttnn::experimental
