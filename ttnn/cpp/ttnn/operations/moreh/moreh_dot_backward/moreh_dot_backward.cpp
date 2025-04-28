// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_dot_backward.hpp"

#include "ttnn/operations/moreh/moreh_dot_backward/device/moreh_dot_backward_device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::moreh::moreh_dot_backward {
std::vector<std::optional<Tensor>> MorehDotBackward::invoke(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& other,
    const std::optional<const Tensor>& input_grad,
    const std::optional<const Tensor>& other_grad,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::moreh_dot_backward(output_grad, input, other, input_grad, other_grad, memory_config);
}

}  // namespace ttnn::operations::moreh::moreh_dot_backward
