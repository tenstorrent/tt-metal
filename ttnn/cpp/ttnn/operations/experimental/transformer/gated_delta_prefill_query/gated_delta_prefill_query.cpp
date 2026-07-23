// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gated_delta_prefill_query.hpp"

#include "device/gated_delta_prefill_query_device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/device.hpp"

namespace ttnn::experimental {

std::tuple<ttnn::Tensor, ttnn::Tensor> gated_delta_prefill_query(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& gate,
    const ttnn::Tensor& decay,
    const ttnn::Tensor& state,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    auto mc = memory_config.value_or(state.memory_config());
    auto kc = init_device_compute_kernel_config(
        state.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi2,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);

    auto out = ttnn::prim::gated_delta_prefill_query(q, k, v, gate, decay, state, mc, kc);
    return {out[0], out[1]};
}

}  // namespace ttnn::experimental
