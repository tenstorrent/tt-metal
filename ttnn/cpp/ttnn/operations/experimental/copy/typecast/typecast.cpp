// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast.hpp"
#include "ttnn/operations/data_movement/copy/device/copy_device_operation.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/to_memory_config/to_memory_config_op.hpp"

namespace ttnn::experimental {

ttnn::Tensor typecast(
    const Tensor& input_tensor,
    const DataType& dtype,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& optional_output_tensor) {
    const auto memory_config = output_mem_config.value_or(input_tensor.memory_config());
    const auto input_dtype = input_tensor.dtype();
    const bool use_copy = input_dtype == dtype || (input_dtype != DataType::FLOAT32 && is_floating_point(input_dtype) &&
                                                   is_floating_point(dtype));
    if (use_copy) {
        return ttnn::prim::copy(input_tensor, memory_config, dtype, optional_output_tensor);
    }
    const auto output_memory_config =
        optional_output_tensor.has_value() ? optional_output_tensor->memory_config() : memory_config;
    const auto& input_memory_config = input_tensor.memory_config();
    if (output_memory_config == input_memory_config ||
        (!output_memory_config.is_sharded() && !input_memory_config.is_sharded())) {
        return ttnn::typecast(input_tensor, dtype, output_memory_config, optional_output_tensor);
    }
    const auto converted = ttnn::typecast(input_tensor, dtype);
    if (optional_output_tensor.has_value()) {
        return ttnn::prim::copy(converted, output_memory_config, dtype, optional_output_tensor);
    }
    return ttnn::to_memory_config(converted, output_memory_config);
}

}  // namespace ttnn::experimental
