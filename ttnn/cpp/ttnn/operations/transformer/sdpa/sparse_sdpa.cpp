// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/sparse_sdpa.hpp"
#include "ttnn/operations/transformer/sdpa/device/sparse_sdpa_device_operation.hpp"
#include "ttnn/operations/core/to_layout/to_layout_op.hpp"
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

    // The device op writes a ROW_MAJOR bf16 result. We expose a TILE-layout output: bf16 q -> bf16 TILE,
    // fp8_e4m3 q -> bfloat8_b TILE (bfloat8_b is block-float, so TILE-only). A SINGLE to_layout does both
    // the row->tile pack and the dtype downcast (tilize takes an output dtype), so there are no extra
    // typecasts and no lossy fp8 intermediate — see PLAN_sparse_sdpa_native_tile_output.md. (to_layout packs
    // row->tile across each head's full [S, v_dim] plane; fusing the tilize into the per-token kernel would
    // need an S<->H transpose, since compute packs heads as the tile-row axis and loops tokens.)
    ttnn::Tensor out =
        ttnn::prim::sparse_sdpa(q, kv, indices, resolved_scale, v_dim, k_chunk_size, kernel_config, cache_batch_idx);

    const ttnn::DataType out_dtype =
        (q.dtype() == ttnn::DataType::FP8_E4M3) ? ttnn::DataType::BFLOAT8_B : ttnn::DataType::BFLOAT16;
    return ttnn::to_layout(out, ttnn::Layout::TILE, out_dtype);
}

}  // namespace ttnn::transformer
