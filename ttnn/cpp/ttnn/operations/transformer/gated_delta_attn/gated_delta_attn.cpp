// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/gated_delta_attn/gated_delta_attn.hpp"
#include "ttnn/operations/transformer/gated_delta_attn/device/gated_delta_attn_device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/device.hpp"

namespace ttnn::transformer {

std::tuple<ttnn::Tensor, ttnn::Tensor> gated_delta_attn_seq(
    const ttnn::Tensor& L_unit,
    const ttnn::Tensor& v_beta_sc,
    const ttnn::Tensor& k_bd_sc,
    const ttnn::Tensor& intra_attn,
    const ttnn::Tensor& q_decay,
    const ttnn::Tensor& k_decay_t,
    const ttnn::Tensor& dl_exp,
    const ttnn::Tensor& L_inv,
    const std::optional<ttnn::Tensor>& initial_state,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    auto mc = memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
    auto kc = init_device_compute_kernel_config(
        L_unit.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi2,
        /*approx=*/false,
        /*fp32_acc=*/true,
        /*packer_l1=*/false);

    auto results = ttnn::prim::gated_delta_attn_seq(
        L_unit, v_beta_sc, k_bd_sc, intra_attn, q_decay, k_decay_t, dl_exp, L_inv, initial_state, mc, kc);

    return {results[0], results[1]};
}

}  // namespace ttnn::transformer
