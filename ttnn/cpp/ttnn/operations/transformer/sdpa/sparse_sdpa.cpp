// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/sparse_sdpa.hpp"
#include "ttnn/operations/transformer/sdpa/device/sparse_sdpa_device_operation.hpp"
#include <tt-metalium/hal.hpp>
#include <cmath>

namespace ttnn::transformer {

ttnn::Tensor sparse_sdpa(
    const ttnn::Tensor& q,
    const ttnn::Tensor& kv,
    const ttnn::Tensor& indices,
    uint32_t v_dim,
    std::optional<float> scale,
    uint32_t k_chunk_size,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<uint32_t> cache_batch_idx) {
    const uint32_t k_dim = q.logical_shape()[3];  // head dim, from the tensor
    const float resolved_scale = scale.value_or(1.0f / std::sqrt(static_cast<float>(k_dim)));

    // fp8 q/kv must be tilized through a 32-bit dest accumulator, so default fp32_dest_acc_en on for fp8.
    const bool any_fp8 = (kv.dtype() == ttnn::DataType::FP8_E4M3) || (q.dtype() == ttnn::DataType::FP8_E4M3);
    auto kernel_config = init_device_compute_kernel_config(
        tt::tt_metal::hal::get_arch(),
        compute_kernel_config,
        /*default_fidelity=*/MathFidelity::HiFi2,
        /*default_approx_mode=*/true,
        /*default_fp32_acc=*/any_fp8,
        /*default_l1_acc=*/false);

    return ttnn::prim::sparse_sdpa(q, kv, indices, resolved_scale, v_dim, k_chunk_size, kernel_config, cache_batch_idx);
}

}  // namespace ttnn::transformer
