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
        std::optional<float> lr,
        std::optional<float> beta1,
        std::optional<float> beta2,
        std::optional<float> eps,
        std::optional<float> weight_decay,
        std::optional<uint32_t> step,
        std::optional<bool> amsgrad,
        const std::optional<const Tensor>& max_exp_avg_sq_in,
        const std::optional<const Tensor>& param_out,
        const std::optional<const Tensor>& exp_avg_out,
        const std::optional<const Tensor>& exp_avg_sq_out,
        const std::optional<const Tensor>& max_exp_avg_sq_out,
        const std::optional<ttnn::MemoryConfig>& memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};
}  // namespace ttnn::operations::moreh::moreh_adam

namespace ttnn {
constexpr auto moreh_adam =
    ttnn::register_operation<"ttnn::moreh_adam", ttnn::operations::moreh::moreh_adam::MorehAdam>();
}
