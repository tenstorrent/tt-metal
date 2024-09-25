// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/moreh_linear_backward/moreh_linear_backward_op.hpp"

#include <tuple>
#include <type_traits>

#include "ttnn/deprecated/tt_dnn/op_library/moreh_matmul/moreh_matmul_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_sum/moreh_sum_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
namespace operations {
namespace primary {

namespace {
std::tuple<bool, bool, bool> get_required_outputs(const std::vector<bool>& are_required_outputs) {
    if (are_required_outputs.size() != 3) {
        TT_ASSERT(are_required_outputs.size() == 3, "are_required_outputs size must be 3");
    }

    return {are_required_outputs[0], are_required_outputs[1], are_required_outputs[2]};
}
}  // namespace

// TODO: Move bias backward code
////////////////////////////////////////////////////////////////////////////
//                         MorehBiasAddBackward
////////////////////////////////////////////////////////////////////////////
void MorehBiasAddBackward::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    const auto& bias_grad = output_tensors.at(0);
    if (bias_grad.has_value() && input_tensors.size() == 2) {
        const auto& bias = input_tensors.at(1);
        const auto& bias_grad_tensor = bias_grad.value();
        TT_ASSERT(is_same_shape(bias, bias_grad_tensor), "both tensors should be the same shape");
        TT_ASSERT(
            is_scalar(bias_grad_tensor) || is_1d_tensor(bias_grad_tensor), "bias_grad tensor should be 1d or scalar");
    }
}

std::vector<ttnn::Shape> MorehBiasAddBackward::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    return {input_tensors.at(1).get_shape().with_tile_padding()};
}

std::vector<Tensor> MorehBiasAddBackward::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.at(0).has_value()) {
        return {output_tensors.at(0).value()};
    }

    TT_FATAL(input_tensors.size() == 2, "Error");
    return operation::generic_create_output_tensors(
        *this, input_tensors, input_tensors.at(1).get_dtype(), Layout::TILE, this->bias_grad_mem_config);
}

operation::ProgramWithCallbacks MorehBiasAddBackward::create_program(
    const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) const {
    const auto& output_grad = inputs.at(0);
    const auto& bias_grad = outputs.at(0);
    return is_scalar(bias_grad) ? (moreh_bias_backward_single_core_hw(output_grad, bias_grad, this->compute_kernel_config))
                                : (moreh_bias_backward_multi_core_h(output_grad, bias_grad, this->compute_kernel_config));
}

inline void moreh_linear_backward_validate(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& weight,
    const std::optional<const Tensor>& input_grad,
    const std::optional<const Tensor>& weight_grad,
    const std::optional<const Tensor>& bias_grad) {
    if (input_grad.has_value()) {
        const auto& input_grad_tensor = input_grad.value();
        TT_ASSERT(is_same_shape(input, input_grad_tensor), "both tensors should be the same shape");
    }

    if (weight_grad.has_value()) {
        const auto& weight_grad_tensor = weight_grad.value();
        TT_ASSERT(is_same_shape(weight, weight_grad_tensor), "both tensors should be the same shape");
    }

    if (bias_grad.has_value()) {
        const auto& bias_grad_tensor = bias_grad.value();
        TT_ASSERT(
            is_scalar(bias_grad_tensor) || is_1d_tensor(bias_grad_tensor), "bias_grad tensor should be 1d or scalar");
    }
}

std::vector<std::optional<Tensor>> moreh_linear_backward(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& weight,
    const std::vector<bool>& are_required_outputs,
    std::optional<const Tensor> bias,
    std::optional<const Tensor> input_grad,
    std::optional<const Tensor> weight_grad,
    std::optional<const Tensor> bias_grad,
    const MemoryConfig& input_grad_mem_config,
    const MemoryConfig& weight_grad_mem_config,
    const MemoryConfig& bias_grad_mem_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    std::vector<std::optional<Tensor>> result(3);
    const auto [input_required_grad, weight_required_grad, bias_required_grad] =
        get_required_outputs(are_required_outputs);

    TT_ASSERT(
        output_grad.storage_type() == StorageType::DEVICE && input.storage_type() == StorageType::DEVICE &&
            weight.storage_type() == StorageType::DEVICE,
        "input and weight tensors need to be on device");

    TT_FATAL(output_grad.storage_type() == StorageType::DEVICE, "Error");
    auto kernel_config_val = init_device_compute_kernel_config(output_grad.device()->arch(), compute_kernel_config, MathFidelity::HiFi4);

    moreh_linear_backward_validate(output_grad, input, weight, input_grad, weight_grad, bias_grad);

    if (input_required_grad) {
        TT_ASSERT(input_grad.has_value(), "input_grad tensor should not be std::nullopt");
        result[0] = moreh_matmul(output_grad, weight, false, false, input_grad, std::nullopt, input_grad_mem_config);
    }

    if (weight_required_grad) {
        TT_ASSERT(weight_grad.has_value(), "weight_grad tensor should not be std::nullopt");
        const auto& weight_grad_tensor = weight_grad.value();
        if (is_same_batch_dim(output_grad, weight_grad_tensor)) {
            moreh_matmul(output_grad, input, true, false, weight_grad_tensor, std::nullopt, weight_grad_mem_config);
        }
        else {
            const auto& temp_weight_grad =
                moreh_matmul(output_grad, input, true, false, std::nullopt, std::nullopt, weight_grad_mem_config);
            TT_ASSERT(weight_grad.has_value(), "weight_grad tensor should not be std::nullopt");
            std::vector<int64_t> dims = find_reduce_dim(temp_weight_grad.get_shape().with_tile_padding(), weight_grad.value().get_shape().with_tile_padding());
            moreh_sum(temp_weight_grad, dims, true, weight_grad.value(), weight_grad_mem_config, compute_kernel_config);
        }
        result[1] = weight_grad_tensor;
    }

    if (bias_required_grad) {
        std::vector<Tensor> input_tensors = { output_grad };
        if (bias) {
            input_tensors.emplace_back(*bias);
        }
        std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({output_grad}))};
        operation::launch_op(
            [bias_grad_mem_config, kernel_config_val](
                const std::vector<Tensor>& input_tensors,
                const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
                return operation::run(
                    MorehBiasAddBackward{.bias_grad_mem_config = bias_grad_mem_config, .compute_kernel_config = kernel_config_val},
                    input_tensors,
                    optional_input_tensors,
                    optional_output_tensors);
            },
            input_tensors,
            output_tensors,
            {},
            {bias_grad});

        result[2] = std::make_optional(output_tensors.at(0));
    }

    return result;
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
