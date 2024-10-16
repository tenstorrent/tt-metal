// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_adam {
struct MorehAdam {
    static std::vector<std::optional<Tensor>> invoke(
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
        const std::optional<const Tensor> max_exp_avg_sq_in,
        const std::optional<const Tensor> param_out,
        const std::optional<const Tensor> exp_avg_out,
        const std::optional<const Tensor> exp_avg_sq_out,
        const std::optional<const Tensor> max_exp_avg_sq_out,
        const std::optional<ttnn::MemoryConfig>& memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);

    static std::vector<Tensor> create_async_output_tensors(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_inputs);

    static std::vector<bool> create_async_return_flag(
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
        const std::optional<const Tensor> max_exp_avg_sq_in,
        const std::optional<const Tensor> param_out,
        const std::optional<const Tensor> exp_avg_out,
        const std::optional<const Tensor> exp_avg_sq_out,
        const std::optional<const Tensor> max_exp_avg_sq_out,
        const std::optional<ttnn::MemoryConfig>& memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};
}  // namespace ttnn::operations::moreh::moreh_adam

namespace ttnn {
constexpr auto moreh_adam =
    ttnn::register_operation_with_auto_launch_op<"ttnn::moreh_adam", ttnn::operations::moreh::moreh_adam::MorehAdam>();
}
