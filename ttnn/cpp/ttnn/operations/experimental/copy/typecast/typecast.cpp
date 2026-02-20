// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operation.hpp"
#include "ttnn/decorators.hpp"
#include "typecast.hpp"
#include "ttnn/operations/data_movement/copy/device/copy_device_operation.hpp"

namespace ttnn::operations::experimental::copy {

ttnn::Tensor TypecastOperation::invoke(
    const Tensor& input_tensor,
    const DataType& dtype,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::prim::copy(
        input_tensor, output_mem_config.value_or(input_tensor.memory_config()), dtype, optional_output_tensor);
}

}  // namespace ttnn::operations::experimental::copy
