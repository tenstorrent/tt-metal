// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "full_like.hpp"
#include "device/full_like_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn {

Tensor moreh_full_like(
    const Tensor& input,
    std::variant<float, int> fill_value,
    const std::optional<DataType>& dtype,
    const std::optional<Layout>& layout,
    const std::optional<MemoryConfig>& memory_config) {
    using OperationType = operations::full_like::FullLikeOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        fill_value,
        dtype.value_or(input.dtype()),
        layout.value_or(input.layout()),
        memory_config.value_or(input.memory_config())};
    auto tensor_args = OperationType::tensor_args_t{input};

    return device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn
