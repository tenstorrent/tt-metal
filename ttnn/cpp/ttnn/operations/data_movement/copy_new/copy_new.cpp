// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "copy_new.hpp"

#include <utility>

#include "ttnn/operations/data_movement/copy_new/device/copy_new_device_operation.hpp"
#include "ttnn/operation.hpp"

using namespace tt::tt_metal;

namespace ttnn {

Tensor copy_new(const Tensor& src_tensor, const Tensor& dst_tensor) {
    return ttnn::prim::copy_new(
        src_tensor, dst_tensor.memory_config(), dst_tensor.dtype(), std::make_optional(dst_tensor));
}

Tensor assign_new(
    const Tensor& input,
    const MemoryConfig& output_mem_config,
    std::optional<const DataType> output_dtype,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::prim::copy_new(input, output_mem_config, output_dtype.value_or(input.dtype()), optional_output_tensor);
}

Tensor assign_new(const Tensor& input_a, const Tensor& input_b) {
    return ttnn::prim::copy_new(input_a, input_b.memory_config(), input_b.dtype(), std::make_optional(input_b));
}

}  // namespace ttnn
