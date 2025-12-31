// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gelu_backward.hpp"
#include "device/gelu_backward_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental {

Tensor gelu_bw(
    const Tensor& grad_output_tensor,
    const Tensor& input_tensor,
    const std::string& approximate,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> input_grad_tensor) {
    using OperationType = operations::experimental::gelu_backward::GeluBackwardDeviceOperation;

    DataType output_dtype = input_tensor.dtype();
    auto output_memory_config = input_grad_tensor.has_value() ? input_grad_tensor.value().memory_config()
                                                              : memory_config.value_or(input_tensor.memory_config());

    auto operation_attributes = OperationType::operation_attributes_t{
        .output_dtype = output_dtype, .output_memory_config = output_memory_config, .approximate = approximate};
    auto tensor_args = OperationType::tensor_args_t{
        .grad_output = grad_output_tensor, .input = input_tensor, .preallocated_input_grad = input_grad_tensor};

    return device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::experimental
