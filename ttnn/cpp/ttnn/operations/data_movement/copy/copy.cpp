// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/copy/copy.hpp"

#include <utility>

#include "device/copy_device_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

ttnn::Tensor CopyOperation::invoke(const Tensor& src_tensor, const Tensor& dst_tensor) {
    return ttnn::prim::copy(src_tensor, dst_tensor.memory_config(), dst_tensor.dtype(), std::make_optional(dst_tensor));
}

ttnn::Tensor AssignOperation::invoke(
    const Tensor& input,
    const MemoryConfig& output_mem_config,
    std::optional<const DataType> output_dtype,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::prim::copy(input, output_mem_config, output_dtype.value_or(input.dtype()), optional_output_tensor);
}

ttnn::Tensor AssignOperation::invoke(
    const Tensor& input, const MemoryConfig& output_mem_config, std::optional<const DataType> output_dtype) {
    return invoke(input, output_mem_config, output_dtype, std::nullopt);
}

ttnn::Tensor AssignOperation::invoke(const Tensor& input_a, const Tensor& input_b) {
    return ttnn::prim::copy(input_a, input_b.memory_config(), input_b.dtype(), std::make_optional(input_b));
}

}  // namespace ttnn::operations::data_movement
