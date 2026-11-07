// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/copy/copy.hpp"

#include <utility>

#include "device/copy_device_operation.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operation.hpp"

using namespace tt::tt_metal;

using ttnn::operations::data_movement::uint16_output_dtype_needs_typecast;

namespace ttnn {

Tensor copy(const Tensor& src_tensor, const Tensor& dst_tensor) {
    // A preallocated output of another dtype makes the copy kernel reinterpret the source's bits,
    // so a uint16 source is converted by value into the caller's own output tensor first.
    if (uint16_output_dtype_needs_typecast(src_tensor.dtype(), dst_tensor.dtype())) {
        return ttnn::typecast(
            src_tensor, dst_tensor.dtype(), dst_tensor.memory_config(), std::make_optional(dst_tensor));
    }
    return ttnn::prim::copy(src_tensor, dst_tensor.memory_config(), dst_tensor.dtype(), std::make_optional(dst_tensor));
}

Tensor assign(
    const Tensor& input,
    const MemoryConfig& output_mem_config,
    std::optional<const DataType> output_dtype,
    const std::optional<Tensor>& optional_output_tensor) {
    if (output_dtype.has_value() && uint16_output_dtype_needs_typecast(input.dtype(), output_dtype.value())) {
        return ttnn::typecast(input, output_dtype.value(), output_mem_config, optional_output_tensor);
    }
    return ttnn::prim::copy(input, output_mem_config, output_dtype.value_or(input.dtype()), optional_output_tensor);
}

Tensor assign(const Tensor& input_a, const Tensor& input_b) {
    if (uint16_output_dtype_needs_typecast(input_a.dtype(), input_b.dtype())) {
        return ttnn::typecast(input_a, input_b.dtype(), input_b.memory_config(), std::make_optional(input_b));
    }
    return ttnn::prim::copy(input_a, input_b.memory_config(), input_b.dtype(), std::make_optional(input_b));
}

}  // namespace ttnn
