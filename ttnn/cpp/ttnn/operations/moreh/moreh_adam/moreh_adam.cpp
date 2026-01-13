// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_adam.hpp"

#include "ttnn/operations/moreh/moreh_adam/device/moreh_adam_device_operation.hpp"
#include "ttnn/operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::moreh::moreh_adam {
std::vector<std::optional<Tensor>> MorehAdam::invoke(
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
    const std::optional<const Tensor>& max_exp_avg_sq_in,
    const std::optional<const Tensor>& param_out,
    const std::optional<const Tensor>& exp_avg_out,
    const std::optional<const Tensor>& exp_avg_sq_out,
    const std::optional<const Tensor>& max_exp_avg_sq_out,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::prim::moreh_adam(
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

}  // namespace ttnn::operations::moreh::moreh_adam
