// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/gated_delta_attn/gated_delta_attn_preprocess.hpp"

#include "ttnn/device.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/gated_delta_attn/device/gated_delta_attn_preprocess_device_operation.hpp"

namespace ttnn::transformer {

std::vector<ttnn::Tensor> gated_delta_attn_preprocess(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& beta,
    const ttnn::Tensor& g,
    const ttnn::Tensor& triu_ones,
    const ttnn::Tensor& tril_mask,
    const ttnn::Tensor& eye,
    const ttnn::Tensor& lower_causal,
    const ttnn::Tensor& eye_32,
    uint32_t chunk_size,
    float diag_alpha,
    bool bf16_value_path,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    auto mc = memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
    auto kc = init_device_compute_kernel_config(
        q.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);

    return ttnn::prim::gated_delta_attn_preprocess(
        q,
        k,
        v,
        beta,
        g,
        triu_ones,
        tril_mask,
        eye,
        lower_causal,
        eye_32,
        chunk_size,
        diag_alpha,
        bf16_value_path,
        mc,
        kc);
}

}  // namespace ttnn::transformer
