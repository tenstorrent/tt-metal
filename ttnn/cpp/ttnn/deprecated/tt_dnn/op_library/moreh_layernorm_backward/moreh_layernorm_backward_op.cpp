// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "ttnn/deprecated/tt_dnn/op_library/moreh_layernorm_backward/moreh_layernorm_backward_op.hpp"

namespace tt {

namespace operations {

namespace primary {

namespace {
inline void check_tensor(const Tensor& tensor, const std::string& op_name) {
    TT_ASSERT(tensor.get_layout() == Layout::TILE, fmt::format("{} only supports tiled layout.", op_name));
    TT_ASSERT(tensor.get_dtype() == DataType::BFLOAT16, fmt::format("{} only supports bfloat16.", op_name));
    TT_ASSERT(
        tensor.storage_type() == StorageType::DEVICE, fmt::format("Operands to {} need to be on device!", op_name));
    TT_ASSERT(
        tensor.buffer() != nullptr, fmt::format("Operands to {} need to be allocated in buffers on device!", op_name));
}
}  // namespace

// input_grad
void MorehLayerNormBackwardInputGrad::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const std::vector<std::optional<Tensor>>& output_tensors
    ) const {
    TT_ASSERT(
        input_tensors.size() == 4 and optional_input_tensors.size() <= 1,
        "moreh_layernorm_backward_input_grad must have between 4 to 5 input tensors");

    const auto& output_grad = input_tensors.at(0);
    const auto& input = input_tensors.at(1);
    const auto& mean = input_tensors.at(2);
    const auto& rstd = input_tensors.at(3);

    const auto& gamma = optional_input_tensors.at(0);

    const auto& input_grad = output_tensors.at(0);

    check_tensor(output_grad, "moreh_layernorm_backward_input_grad");
    check_tensor(input, "moreh_layernorm_backward_input_grad");
    check_tensor(mean, "moreh_layernorm_backward_input_grad");
    check_tensor(rstd, "moreh_layernorm_backward_input_grad");
    if (input_grad.has_value()) {
        check_tensor(input_grad.value(), "moreh_layernorm_backward_input_grad");
    }

    TT_ASSERT(this->normalized_dims > 0);
    TT_ASSERT(this->normalized_dims <= output_grad.get_legacy_shape().rank());

    if (gamma.has_value()) {
        check_tensor(gamma.value(), "moreh_layernorm_backward_input_grad");
    }
}

std::vector<Shape> MorehLayerNormBackwardInputGrad::compute_output_shapes(
    const std::vector<Tensor>& input_tensors) const {
    auto input = input_tensors.at(0);
    auto input_shape = input.get_legacy_shape();

    // The shapes of the input and output are always the same.
    return {input_shape};
}

std::vector<Tensor> MorehLayerNormBackwardInputGrad::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.at(0).has_value()) {
        log_debug(LogOp, "{}:{} use output tensor", __func__, __LINE__);
        return {output_tensors.at(0).value()};
    }
    TT_FATAL(false, "Create output tensor is not supported yet. fix this after the # 9552 issue is addressed.");
    return {};
}

operation::ProgramWithCallbacks MorehLayerNormBackwardInputGrad::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& output_grad = input_tensors.at(0);
    const auto& input = input_tensors.at(1);
    const auto& mean = input_tensors.at(2);
    const auto& rstd = input_tensors.at(3);

    const auto& gamma = optional_input_tensors.at(0);

    const auto& input_grad = output_tensors.at(0);

    return moreh_layernorm_backward_input_grad_impl(
        output_grad, input, mean, rstd, this->normalized_dims, input_grad, this->compute_kernel_config, gamma);
}

// gamma_grad and beta_grad
void MorehLayerNormBackwardGammaBetaGrad::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<Tensor>>& output_tensors) const {

    TT_ASSERT(
        input_tensors.size() == 4 and output_tensors.size() <= 2,
        "moreh_layernorm_backward_gamma_beta_grad must have between 4 to 6 input tensors");

    const auto& output_grad = input_tensors.at(0);
    const auto& input = input_tensors.at(1);
    const auto& mean = input_tensors.at(2);
    const auto& rstd = input_tensors.at(3);

    check_tensor(output_grad, "moreh_layernorm_backward_gamma_beta_grad");
    check_tensor(input, "moreh_layernorm_backward_gamma_beta_grad");
    check_tensor(mean, "moreh_layernorm_backward_gamma_beta_grad");
    check_tensor(rstd, "moreh_layernorm_backward_gamma_beta_grad");

    TT_ASSERT(this->normalized_dims > 0);
    TT_ASSERT(this->normalized_dims <= output_grad.get_legacy_shape().rank());

    const auto& gamma_grad = output_tensors.at(0);
    const auto& beta_grad = output_tensors.at(1);
    if (gamma_grad.has_value()) {
        check_tensor(gamma_grad.value(), "moreh_layernorm_backward_gamma_beta_grad");
    }

    if (beta_grad.has_value()) {
        check_tensor(beta_grad.value(), "moreh_layernorm_backward_gamma_beta_grad");
    }
}

std::vector<Shape> MorehLayerNormBackwardGammaBetaGrad::compute_output_shapes(
    const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(false, "The compute_output_shapes function in MorehLayerNormBackwardGammaBetaGrad is not implemented.");
    return {};
}

std::vector<Tensor> MorehLayerNormBackwardGammaBetaGrad::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.at(0).has_value() && output_tensors.at(1).has_value()) {
        log_debug(LogOp, "{}:{} use output tensor", __func__, __LINE__);
        return {output_tensors.at(0).value(), output_tensors.at(1).value()};
    }

    TT_FATAL(false, "Create output tensor is not supported yet. Fix this after the #9552 issue is addressed.");
    return {};
}

operation::ProgramWithCallbacks MorehLayerNormBackwardGammaBetaGrad::create_program(
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& output_grad = input_tensors.at(0);
    const auto& input = input_tensors.at(1);
    const auto& mean = input_tensors.at(2);
    const auto& rstd = input_tensors.at(3);

    auto& gamma_grad = output_tensors.at(0);
    auto& beta_grad = output_tensors.at(1);

    return moreh_layernorm_backward_gamma_beta_grad_impl(
        output_grad, input, mean, rstd, this->normalized_dims, this->compute_kernel_config, gamma_grad, beta_grad);
}

// input_grad
Tensor moreh_layernorm_backward_input_grad(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    uint32_t normalized_dims,
    const std::optional<const Tensor> input_grad,
    const std::optional<const Tensor> gamma,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {

    auto device = input.device();
    auto compute_kernel_config_val =
        init_device_compute_kernel_config(DeviceArch(device), compute_kernel_config, MathFidelity::HiFi4);

    std::vector<Tensor> output_tensors = {
        Tensor(operation::get_workers_for_op_output({output_grad, input, mean, rstd}, {gamma}))};
    operation::launch_op(
        [normalized_dims, memory_config, compute_kernel_config_val](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                MorehLayerNormBackwardInputGrad{
                    .normalized_dims = normalized_dims, .memory_config = memory_config.value_or(input_tensors.at(0).memory_config()),
                    .compute_kernel_config = compute_kernel_config_val},
                input_tensors,
                optional_input_tensors,
                optional_output_tensors);
        },
        {output_grad, input, mean, rstd},
        output_tensors,
        {gamma},
        {input_grad});

    return output_tensors.at(0);
}

// gamma_grad and beta_grad
std::vector<std::optional<Tensor>> moreh_layernorm_backward_gamma_beta_grad(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    uint32_t normalized_dims,
    const std::optional<const Tensor> gamma_grad,
    const std::optional<const Tensor> beta_grad,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {

    auto device = input.device();
    auto compute_kernel_config_val =
        init_device_compute_kernel_config(DeviceArch(device), compute_kernel_config, MathFidelity::HiFi4);

    std::vector<std::optional<Tensor>> outputs(2);
    if (!gamma_grad.has_value() && !beta_grad.has_value()) {
        return outputs;
    }

    std::vector<Tensor> dummy_output_tensors = {};
    if (gamma_grad.has_value()) {
        dummy_output_tensors.push_back(Tensor(operation::get_workers_for_op_output({output_grad, input, mean, rstd}, {})));
    }

    if (gamma_grad.has_value()) {
        dummy_output_tensors.push_back(Tensor(operation::get_workers_for_op_output({output_grad, input, mean, rstd}, {})));
    }

    operation::launch_op(
        [normalized_dims, memory_config, compute_kernel_config_val](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                MorehLayerNormBackwardGammaBetaGrad{
                    .normalized_dims = normalized_dims, .memory_config = memory_config.value_or(input_tensors.at(0).memory_config()),
                    .compute_kernel_config = compute_kernel_config_val},
                input_tensors,
                optional_input_tensors,
                optional_output_tensors);
        },
        {output_grad, input, mean, rstd},
        dummy_output_tensors,
        {},
        {gamma_grad, beta_grad});

    if (gamma_grad.has_value()) {
        outputs[0] = gamma_grad.value();
    }

    if (beta_grad.has_value()) {
        outputs[1] = beta_grad.value();
    }
    return outputs;
}

// input_grad and gamma_grad and beta_grad
std::vector<std::optional<Tensor>> moreh_layernorm_backward(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    uint32_t normalized_dims,
    const std::optional<const Tensor> gamma,
    const std::optional<const Tensor> input_grad,
    const std::optional<const Tensor> gamma_grad,
    const std::optional<const Tensor> beta_grad,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    std::vector<std::optional<Tensor>> outputs;
    outputs.reserve(3);

    // input_grad
    if (input_grad.has_value()) {
        outputs.push_back(moreh_layernorm_backward_input_grad(
            output_grad, input, mean, rstd, normalized_dims, input_grad.value(), gamma, memory_config, compute_kernel_config));
    } else {
        outputs.push_back(std::nullopt);
    }

    // gamma_grad and beta_grad
    const auto& gamma_beta_grad = moreh_layernorm_backward_gamma_beta_grad(
        output_grad, input, mean, rstd, normalized_dims, gamma_grad, beta_grad, memory_config, compute_kernel_config);
    outputs.push_back(gamma_beta_grad[0]);
    outputs.push_back(gamma_beta_grad[1]);

    return outputs;
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
