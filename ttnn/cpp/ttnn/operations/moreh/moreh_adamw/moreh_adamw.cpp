// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_adamw.hpp"

#include "ttnn/operations/moreh/moreh_adamw/device/moreh_adamw_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_adamw {

std::vector<std::optional<Tensor>> MorehAdamw::invoke(
    const Tensor& param_in,
    const Tensor& grad,
    const Tensor& exp_avg_in,
    const Tensor& exp_avg_sq_in,

    const std::optional<float> lr,
    const std::optional<float> beta1,
    const std::optional<float> beta2,
    const std::optional<float> eps,
    const std::optional<float> weight_decay,
    const std::optional<uint32_t> step,
    const std::optional<bool> amsgrad,

    const std::optional<Tensor>& max_exp_avg_sq_in,
    const std::optional<Tensor>& param_out,
    const std::optional<Tensor>& exp_avg_out,
    const std::optional<Tensor>& exp_avg_sq_out,
    const std::optional<Tensor>& max_exp_avg_sq_out,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    return ttnn::prim::moreh_adamw(
        param_in,
        grad,
        exp_avg_in,
        exp_avg_sq_in,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        step,
        amsgrad,
        max_exp_avg_sq_in,
        param_out,
        exp_avg_out,
        exp_avg_sq_out,
        max_exp_avg_sq_out,
        memory_config,
        compute_kernel_config);
}

std::vector<Tensor> MorehAdamw::create_async_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs) {
    const auto& input_tensor = input_tensors.at(0);
    return {
        Tensor(operation::get_workers_for_op_output({input_tensor})),
        Tensor(operation::get_workers_for_op_output({input_tensor})),
        Tensor(operation::get_workers_for_op_output({input_tensor})),
        Tensor(operation::get_workers_for_op_output({input_tensor}))};
}

std::vector<bool> MorehAdamw::create_async_return_flag(
    const Tensor& param_in,
    const Tensor& grad,
    const Tensor& exp_avg_in,
    const Tensor& exp_avg_sq_in,

    const std::optional<float> lr,
    const std::optional<float> beta1,
    const std::optional<float> beta2,
    const std::optional<float> eps,
    const std::optional<float> weight_decay,
    const std::optional<uint32_t> step,
    const std::optional<bool> amsgrad,

    const std::optional<Tensor>& max_exp_avg_sq_in,
    const std::optional<Tensor>& param_out,
    const std::optional<Tensor>& exp_avg_out,
    const std::optional<Tensor>& exp_avg_sq_out,
    const std::optional<Tensor>& max_exp_avg_sq_out,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    return std::vector<bool>{true, true, true, amsgrad.has_value() ? amsgrad.value() : false};
}
}  // namespace ttnn::operations::moreh::moreh_adamw
