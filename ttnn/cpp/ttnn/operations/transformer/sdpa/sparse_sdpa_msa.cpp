// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/sparse_sdpa_msa.hpp"
#include "ttnn/operations/transformer/sdpa/device/sparse_sdpa_msa_device_operation.hpp"
#include <tt-metalium/hal.hpp>
#include <cmath>

namespace ttnn::transformer {

ttnn::Tensor sparse_sdpa_msa(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& indices,
    std::optional<float> scale,
    uint32_t block_size,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<uint32_t> cache_batch_idx,
    std::optional<uint32_t> chunk_start_idx,
    std::optional<uint32_t> cluster_axis) {
    const uint32_t d = q.logical_shape()[3];  // head dim, from the tensor
    const float resolved_scale = scale.value_or(1.0f / std::sqrt(static_cast<float>(d)));

    // fp8 Q needs 32-bit DEST for tilize; bf16 Q uses the default DEST width.
    const bool q_is_fp8 = (q.dtype() == ttnn::DataType::FP8_E4M3);
    auto kernel_config = init_device_compute_kernel_config(
        tt::tt_metal::hal::get_arch(),
        compute_kernel_config,
        /*default_fidelity=*/MathFidelity::HiFi2,
        /*default_approx_mode=*/true,
        /*default_fp32_acc=*/q_is_fp8,
        /*default_l1_acc=*/false);

    return ttnn::prim::sparse_sdpa_msa(
        q, k, v, indices, resolved_scale, block_size, kernel_config, cache_batch_idx, chunk_start_idx, cluster_axis);
}

}  // namespace ttnn::transformer
