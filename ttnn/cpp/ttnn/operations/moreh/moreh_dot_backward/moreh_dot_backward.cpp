// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_dot_backward.hpp"

#include "ttnn/operations/moreh/moreh_dot_backward/device/moreh_dot_backward_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_dot_backward {
std::vector<std::optional<Tensor>> MorehDotBackward::invoke(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &other,
    const std::optional<const Tensor> &input_grad,
    const std::optional<const Tensor> &other_grad,
    const std::optional<MemoryConfig> &memory_config) {
    return ttnn::prim::moreh_dot_backward(output_grad, input, other, input_grad, other_grad, memory_config);
}

std::vector<Tensor> MorehDotBackward::create_async_output_tensors(
    const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>> &optional_inputs) {
    auto output_grad = input_tensors.at(0);
    auto input = input_tensors.at(1);
    auto other = input_tensors.at(2);

    return {
        Tensor(operation::get_workers_for_op_output({output_grad, input, other})),
        Tensor(operation::get_workers_for_op_output({output_grad, input, other})),
    };
}

std::vector<bool> MorehDotBackward::create_async_return_flag(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &other,
    const std::optional<const Tensor> &input_grad,
    const std::optional<const Tensor> &other_grad,
    const std::optional<MemoryConfig> &memory_config) {
    return {input_grad.has_value(), other_grad.has_value()};
}

}  // namespace ttnn::operations::moreh::moreh_dot_backward
