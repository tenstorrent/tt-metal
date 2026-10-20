// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_decode.hpp"

#include "device/kda_decode_device_operation.hpp"
#include "ttnn/device.hpp"

using namespace tt::tt_metal;

namespace ttnn::transformer {

ttnn::Tensor kda_decode(
    const ttnn::Tensor& qkv,
    const ttnn::Tensor& gates,
    const ttnn::Tensor& conv_state,
    const ttnn::Tensor& recurrent_state,
    const ttnn::Tensor& taps,
    const ttnn::Tensor& chan_consts,
    const ttnn::Tensor& o_norm_weight,
    const ttnn::Tensor& selectors,
    uint32_t num_heads,
    float gate_lower_bound,
    float l2_eps,
    float norm_eps,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    auto* dev = qkv.device();
    const auto out_mem = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    const auto kernel_cfg = init_device_compute_kernel_config(
        dev->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);
    auto res = ttnn::prim::kda_decode(
        qkv,
        gates,
        conv_state,
        recurrent_state,
        taps,
        chan_consts,
        o_norm_weight,
        selectors,
        num_heads,
        gate_lower_bound,
        l2_eps,
        norm_eps,
        out_mem,
        kernel_cfg);
    return res[0];
}

}  // namespace ttnn::transformer
