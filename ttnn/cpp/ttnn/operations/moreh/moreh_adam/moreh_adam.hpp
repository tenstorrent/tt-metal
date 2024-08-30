// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>

#include "ttnn/decorators.hpp"
#include "ttnn/operations/moreh/moreh_adam/device/moreh_adam_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_adam {
struct MorehAdam {
    static std::vector<std::optional<Tensor>> invoke(
        const Tensor& param_in,
        const Tensor& grad,
        const Tensor& exp_avg_in,
        const Tensor& exp_avg_sq_in,
        float lr,
        float beta1,
        float beta2,
        float eps,
        float weight_decay,
        uint32_t step,
        bool amsgrad,
        const std::optional<const Tensor> max_exp_avg_sq_in,
        const std::optional<const Tensor> param_out,
        const std::optional<const Tensor> exp_avg_out,
        const std::optional<const Tensor> exp_avg_sq_out,
        const std::optional<const Tensor> max_exp_avg_sq_out,
        const MemoryConfig& memory_config,
        const DeviceComputeKernelConfig compute_kernel_config);
};
}  // namespace ttnn::operations::moreh::moreh_adam

namespace ttnn {
constexpr auto moreh_adam =
    ttnn::register_operation<"ttnn::moreh_adam", ttnn::operations::moreh::moreh_adam::MorehAdam>();
}
