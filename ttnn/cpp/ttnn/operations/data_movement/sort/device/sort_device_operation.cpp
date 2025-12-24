// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sort_device_operation.hpp"
#include "ttnn/device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {
ttnn::operations::data_movement::sort::SortDeviceOperation::tensor_return_value_t sort(
    const Tensor& input_tensor,
    int8_t dim,
    bool descending,
    bool stable,
    const MemoryConfig& output_memory_config,
    const std::vector<std::optional<Tensor>>& output_tensors) {
    using OperationType = ttnn::operations::data_movement::sort::SortDeviceOperation;
    return ttnn::device_operation::detail::launch_on_device<OperationType>(
        OperationType::operation_attributes_t{dim, descending, stable, output_memory_config},
        OperationType::tensor_args_t{input_tensor, output_tensors});
}
}  // namespace ttnn::prim

namespace ttnn::operations::data_movement::sort {

}  // namespace ttnn::operations::data_movement::sort
