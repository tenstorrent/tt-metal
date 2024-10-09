// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_matmul_backward.hpp"

#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/operations/moreh/moreh_dot_op_backward/moreh_dot_backward.hpp"
#include "ttnn/operations/moreh/moreh_matmul/device/moreh_matmul_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_matmul/moreh_matmul.hpp"
#include "ttnn/operations/moreh/moreh_sum/moreh_sum.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::moreh::moreh_matmul_backward {

inline bool is_dot_backward(const Tensor& output_grad, const Tensor& input, const Tensor& other) {
    // TODO: non-4d support for dot backward.
    if (output_grad.get_legacy_shape().rank() != 4 || input.get_legacy_shape().rank() != 4 ||
        other.get_legacy_shape().rank() != 4) {
        return false;
    }
    return tt::operations::primary::is_scalar(output_grad) && tt::operations::primary::is_1d_tensor(input) &&
           tt::operations::primary::is_1d_tensor(other) && tt::operations::primary::is_same_shape(input, other);
}

std::vector<std::optional<Tensor>> MorehMatmulBackward::invoke(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& other,
    const std::vector<bool>& are_required_outputs,
    const std::optional<const Tensor>& input_grad,
    const std::optional<const Tensor>& other_grad,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    if (is_dot_backward(output_grad, input, other)) {
        return ttnn::moreh_dot_backward(output_grad, input, other, input_grad, other_grad, memory_config);
    }

    std::vector<std::optional<Tensor>> outputs(2);

    const bool input_requires_grad = are_required_outputs.at(0);
    const bool other_requires_grad = are_required_outputs.at(1);

    if (input_requires_grad) {
        TT_FATAL(input_grad.has_value(), "Input gradient is marked required but not provided.");
        const auto& input_grad_tensor = input_grad.value();
        if (moreh_matmul::is_same_batch_dim(output_grad, input_grad_tensor)) {
            const auto& input_grad_shape = input_grad_tensor.get_legacy_shape().without_padding();
            const auto& output_grad_shape = output_grad.get_legacy_shape().without_padding();
            ttnn::moreh_matmul(
                output_grad, other, false, true, input_grad_tensor, std::nullopt, memory_config, compute_kernel_config);
        } else {
            const auto& input_shape = input.get_legacy_shape().without_padding();
            const auto& temp_input_grad = ttnn::moreh_matmul(
                output_grad, other, false, true, std::nullopt, std::nullopt, memory_config, compute_kernel_config);
            auto reduce_dims =
                moreh_matmul::find_reduce_dim(temp_input_grad.get_legacy_shape(), input_grad_tensor.get_legacy_shape());
            ttnn::moreh_sum(
                temp_input_grad, reduce_dims, true, input_grad_tensor, memory_config, compute_kernel_config);
        }
        outputs[0] = input_grad_tensor;
    }

    if (other_requires_grad) {
        TT_FATAL(other_grad.has_value(), "Other gradient is marked required but not provided.");
        const auto& other_grad_tensor = other_grad.value();
        if (moreh_matmul::is_same_batch_dim(output_grad, other_grad_tensor)) {
            ttnn::moreh_matmul(
                input, output_grad, true, false, other_grad_tensor, std::nullopt, memory_config, compute_kernel_config);
        } else {
            const auto& temp_other_grad = ttnn::moreh_matmul(
                input, output_grad, true, false, std::nullopt, std::nullopt, memory_config, compute_kernel_config);
            auto reduce_dims =
                moreh_matmul::find_reduce_dim(temp_other_grad.get_legacy_shape(), other_grad_tensor.get_legacy_shape());
            ttnn::moreh_sum(
                temp_other_grad, reduce_dims, true, other_grad_tensor, memory_config, compute_kernel_config);
        }
        outputs[1] = other_grad_tensor;
    }

    return outputs;
}

std::vector<Tensor> MorehMatmulBackward::create_async_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs) {
    const auto& output_grad = input_tensors.at(0);
    const auto& input = input_tensors.at(1);
    const auto& other = input_tensors.at(2);

    return {
        Tensor(operation::get_workers_for_op_output({output_grad, input, other})),
        Tensor(operation::get_workers_for_op_output({output_grad, input, other})),
    };
}

std::vector<bool> MorehMatmulBackward::create_async_return_flag(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& other,
    const std::vector<bool>& are_required_outputs,
    const std::optional<const Tensor>& input_grad,
    const std::optional<const Tensor>& other_grad,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    return {are_required_outputs.at(0), are_required_outputs.at(1)};
}

}  // namespace ttnn::operations::moreh::moreh_matmul_backward
