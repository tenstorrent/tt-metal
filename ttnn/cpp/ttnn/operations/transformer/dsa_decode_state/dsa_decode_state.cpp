// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "dsa_decode_state.hpp"

#include "device/dsa_decode_state_device_operation.hpp"
#include "ttnn/device.hpp"

using namespace tt::tt_metal;

namespace ttnn::transformer {

ttnn::Tensor dsa_decode_state(
    const ttnn::Tensor& stem,
    const ttnn::Tensor& latent_cache,
    const ttnn::Tensor& latent_page_table,
    const ttnn::Tensor& positions,
    const ttnn::Tensor& ring_k,
    const ttnn::Tensor& ring_g,
    const ttnn::Tensor& pool_cache,
    const ttnn::Tensor& qa_norm_w,
    const ttnn::Tensor& kva_norm_w,
    const ttnn::Tensor& k_norm_w,
    const ttnn::Tensor& k_norm_b,
    const ttnn::Tensor& ape,
    const ttnn::Tensor& selectors,
    uint32_t qa_offset,
    uint32_t kva_offset,
    uint32_t k_offset,
    uint32_t g_offset,
    uint32_t q_lora_rank,
    uint32_t latent_dim,
    uint32_t index_dim,
    uint32_t pool_size,
    uint32_t pool_blocks,
    float eps_rms,
    float eps_ln,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    auto* dev = stem.device();
    const auto out_mem = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    const auto kernel_cfg = init_device_compute_kernel_config(
        dev->arch(), compute_kernel_config, MathFidelity::HiFi4, /*approx=*/false, /*fp32_acc=*/true, /*l1_acc=*/false);
    ttnn::prim::DsaDecodeStateInputs inputs{
        .stem = stem,
        .latent_cache = latent_cache,
        .latent_page_table = latent_page_table,
        .positions = positions,
        .ring_k = ring_k,
        .ring_g = ring_g,
        .pool_cache = pool_cache,
        .qa_norm_w = qa_norm_w,
        .kva_norm_w = kva_norm_w,
        .k_norm_w = k_norm_w,
        .k_norm_b = k_norm_b,
        .ape = ape,
        .selectors = selectors,
    };
    auto res = ttnn::prim::dsa_decode_state(
        inputs, qa_offset, kva_offset, k_offset, g_offset, q_lora_rank, latent_dim, index_dim, pool_size, pool_blocks,
        eps_rms, eps_ln, out_mem, kernel_cfg);
    return res[0];
}

}  // namespace ttnn::transformer
