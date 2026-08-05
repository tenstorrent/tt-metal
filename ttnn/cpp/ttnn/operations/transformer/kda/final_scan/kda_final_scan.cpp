// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_final_scan.hpp"

#include "device/kda_final_scan_device_operation.hpp"

namespace ttnn::transformer {

std::vector<ttnn::Tensor> kda_final_chunk_scan(
    const ttnn::Tensor& v_beta,
    const ttnn::Tensor& kd,
    const ttnn::Tensor& q_decay,
    const ttnn::Tensor& intra,
    const ttnn::Tensor& k_dec_t,
    const ttnn::Tensor& final_decay,
    const ttnn::Tensor& t_inv,
    const std::optional<ttnn::Tensor>& initial_state,
    uint32_t chunk_size,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    bool state_only,
    const std::optional<ttnn::Tensor>& identity_tile,
    bool summary_pair,
    bool output_bf16) {
    const auto output_memory_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    const auto kernel_config = init_device_compute_kernel_config(
        v_beta.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);
    return ttnn::prim::kda_final_chunk_scan(
        v_beta,
        kd,
        q_decay,
        intra,
        k_dec_t,
        final_decay,
        t_inv,
        initial_state,
        chunk_size,
        output_memory_config,
        kernel_config,
        state_only,
        identity_tile,
        summary_pair,
        output_bf16);
}

}  // namespace ttnn::transformer
