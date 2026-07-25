// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "triangle_solve.hpp"

#include "device/triangle_solve_device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/device.hpp"

namespace ttnn::experimental {

ttnn::Tensor triangle_solve(
    const ttnn::Tensor& l_neg,
    const ttnn::Tensor& rhs,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    auto mc = memory_config.value_or(rhs.memory_config());
    auto kc = init_device_compute_kernel_config(
        rhs.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi2,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/false,
        /*default_l1_acc=*/false);

    auto out = ttnn::prim::triangle_solve(l_neg, rhs, mc, kc);
    return out[0];
}

}  // namespace ttnn::experimental
