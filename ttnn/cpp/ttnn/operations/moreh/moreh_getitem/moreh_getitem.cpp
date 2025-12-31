// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_getitem.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::operations::moreh::moreh_getitem {

Tensor moreh_getitem(
    const std::optional<const Tensor>& input,
    const std::vector<Tensor>& index_tensors,
    const ttnn::SmallVector<uint32_t>& index_dims,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config) {
    if (!input.has_value()) {
        // FIXME: This is a hack to work around limitations in the decorator
        // infra which requires either an input tensor or a vector of input
        // tensors but not both; wrapping the input tensor in an optional allows
        // us to work around this without rewriting half of the runtime.
        TT_THROW("Input tensor is required for moreh_getitem operation.");
    }
    using OperationType = MorehGetItemOperation;
    auto operation_attributes =
        OperationType::operation_attributes_t{index_dims, memory_config.value_or(input->memory_config())};
    auto tensor_args = OperationType::tensor_args_t{input.value(), index_tensors, output};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::operations::moreh::moreh_getitem
