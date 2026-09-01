// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/vsa_sdpa.hpp"
#include "ttnn/operations/transformer/sdpa/device/vsa_sdpa_device_operation.hpp"
#include <tt-metalium/hal.hpp>
#include <cmath>

namespace ttnn::transformer {

ttnn::Tensor vsa_sdpa(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& indices,
    const ttnn::Tensor& block_counts,
    std::optional<float> scale,
    uint32_t block_size,
    uint32_t k_chunk_blocks,
    bool streaming,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    const uint32_t d = q.logical_shape()[3];  // head dim, from the tensor
    const float resolved_scale = scale.value_or(1.0f / std::sqrt(static_cast<float>(d)));

    auto kernel_config = init_device_compute_kernel_config(
        tt::tt_metal::hal::get_arch(),
        compute_kernel_config,
        /*default_fidelity=*/MathFidelity::HiFi2,
        /*default_approx_mode=*/true,
        /*default_fp32_acc=*/false,
        /*default_l1_acc=*/false);

    return ttnn::prim::vsa_sdpa(
        q, k, v, indices, block_counts, resolved_scale, block_size, k_chunk_blocks, streaming, kernel_config);
}

}  // namespace ttnn::transformer
