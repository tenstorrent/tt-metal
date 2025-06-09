// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/gelu_backward_device_operation.hpp"
#include "gelu_backward.hpp"

namespace ttnn::operations::experimental {

Tensor GeluBackwardOperation::invoke(
    QueueId queue_id,
    const Tensor& grad_output_tensor,
    const Tensor& input_tensor,
    const string& approximate,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> input_grad_tensor) {
    DataType output_dtype = input_tensor.dtype();
    auto arch = input_tensor.device()->arch();
    auto output_memory_config = input_grad_tensor.has_value() ? input_grad_tensor.value().memory_config()
                                                              : memory_config.value_or(input_tensor.memory_config());

    return ttnn::prim::gelu_bw(
        queue_id, grad_output_tensor, input_tensor, approximate, output_dtype, output_memory_config, input_grad_tensor);
}
}  // namespace ttnn::operations::experimental
