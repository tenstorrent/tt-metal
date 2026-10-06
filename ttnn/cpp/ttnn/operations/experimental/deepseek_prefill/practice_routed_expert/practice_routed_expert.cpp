// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "practice_routed_expert.hpp"

#include "device/practice_routed_expert_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::practice_routed_expert {

ttnn::Tensor practice_routed_expert(
    const ttnn::Tensor& x,
    const ttnn::Tensor& w_gate,
    const ttnn::Tensor& w_up,
    const ttnn::Tensor& w_down,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    // Precise defaults rather than production's LoFi: the kernel runs each whole K reduction in DEST.
    const auto kernel_config = init_device_compute_kernel_config(
        x.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true);
    return ttnn::prim::practice_routed_expert(x, w_gate, w_up, w_down, kernel_config);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::practice_routed_expert
