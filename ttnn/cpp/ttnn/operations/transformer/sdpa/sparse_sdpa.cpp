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
    std::optional<uint32_t> cache_batch_idx,
    std::optional<uint32_t> block_cyclic_sp,
    std::optional<uint32_t> block_cyclic_chunk) {
    const uint32_t k_dim = q.logical_shape()[3];  // head dim, from the tensor
    const float resolved_scale = scale.value_or(1.0f / std::sqrt(static_cast<float>(k_dim)));

    // block-cyclic remap: both sp and chunk must be provided together (or neither).
    TT_FATAL(
        block_cyclic_sp.has_value() == block_cyclic_chunk.has_value(),
        "sparse_sdpa: block_cyclic_sp and block_cyclic_chunk must both be set or both unset");
    std::optional<ttnn::prim::BlockCyclicLayout> block_cyclic = std::nullopt;
    if (block_cyclic_sp.has_value()) {
        block_cyclic = ttnn::prim::BlockCyclicLayout{block_cyclic_sp.value(), block_cyclic_chunk.value()};
    }

    // fp8 q/kv must be tilized through a 32-bit dest accumulator, so default fp32_dest_acc_en on for fp8.
    const bool any_fp8 = (kv.dtype() == ttnn::DataType::FP8_E4M3) || (q.dtype() == ttnn::DataType::FP8_E4M3);
    auto kernel_config = init_device_compute_kernel_config(
        tt::tt_metal::hal::get_arch(),
        compute_kernel_config,
        /*default_fidelity=*/MathFidelity::HiFi2,
        /*default_approx_mode=*/true,
        /*default_fp32_acc=*/any_fp8,
        /*default_l1_acc=*/false);

    return ttnn::prim::sparse_sdpa(
        q, kv, indices, resolved_scale, v_dim, k_chunk_size, kernel_config, cache_batch_idx, block_cyclic);
}

}  // namespace ttnn::transformer
