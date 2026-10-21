// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/deepseek_v4_prefill_sdpa.hpp"
#include "ttnn/operations/transformer/sdpa/device/deepseek_v4_prefill_sdpa_device_operation.hpp"
#include <tt-metalium/hal.hpp>
#include <cmath>

namespace ttnn::transformer {

ttnn::Tensor deepseek_v4_prefill_sdpa(
    const ttnn::Tensor& q,
    const ttnn::Tensor& window_kv,
    const ttnn::Tensor& start,
    const std::optional<ttnn::Tensor>& cache_kv,
    const std::optional<ttnn::Tensor>& cache_idx,
    uint32_t window_size,
    std::optional<uint32_t> v_dim,
    std::optional<float> scale,
    uint32_t k_chunk_size,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<ttnn::Tensor>& attention_sink,
    uint32_t compress_ratio,
    bool empty_tail) {
    const uint32_t k_dim = q.logical_shape()[3];
    auto kernel_config = init_device_compute_kernel_config(
        tt::tt_metal::hal::get_arch(),
        compute_kernel_config,
        /*default_fidelity=*/MathFidelity::HiFi2,
        /*default_approx_mode=*/true,
        /*default_fp32_acc=*/false,
        /*default_l1_acc=*/false);
    return ttnn::prim::deepseek_v4_prefill_sdpa(
        q,
        window_kv,
        start,
        cache_kv,
        cache_idx,
        scale.value_or(1.0f / std::sqrt(static_cast<float>(k_dim))),
        v_dim.value_or(k_dim),
        window_size,
        k_chunk_size,
        compress_ratio,
        empty_tail,
        kernel_config,
        attention_sink);
}

}  // namespace ttnn::transformer
