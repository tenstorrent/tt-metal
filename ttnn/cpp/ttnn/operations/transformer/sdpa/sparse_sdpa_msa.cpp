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
    std::optional<uint32_t> cluster_axis,
    std::optional<uint32_t> block_cyclic_sp_axis,
    std::optional<uint32_t> block_cyclic_chunk_local) {
    const uint32_t d = q.logical_shape()[3];  // head dim, from the tensor
    const float resolved_scale = scale.value_or(1.0f / std::sqrt(static_cast<float>(d)));

    // Block-cyclic cache: resolve sp from the mesh (a caller can't pass an sp that disagrees with the device)
    // and cross-check chunk_local against q, so the invP remap gets ground-truth layout, not a trusted arg.
    TT_FATAL(
        block_cyclic_sp_axis.has_value() == block_cyclic_chunk_local.has_value(),
        "block_cyclic_sp_axis and block_cyclic_chunk_local must be set together");
    std::optional<ttnn::prim::BlockCyclicLayout> block_cyclic;
    if (block_cyclic_sp_axis.has_value()) {
        const auto mesh_shape = q.device()->get_view().shape();
        const uint32_t sp_axis = block_cyclic_sp_axis.value();
        TT_FATAL(
            sp_axis < mesh_shape.dims(),
            "block_cyclic_sp_axis ({}) out of range for mesh dims ({})",
            sp_axis,
            mesh_shape.dims());
        const uint32_t sp = mesh_shape[sp_axis];
        const uint32_t chunk_local = block_cyclic_chunk_local.value();
        const uint32_t tp = mesh_shape.mesh_size() / sp;
        const uint32_t q_isl = q.logical_shape()[2];
        TT_FATAL(
            chunk_local == q_isl || chunk_local == q_isl * tp,
            "block_cyclic_chunk_local ({}) must equal q seq-len ({}) or tp*q seq-len ({})",
            chunk_local,
            q_isl,
            q_isl * tp);
        block_cyclic = ttnn::prim::BlockCyclicLayout{sp, chunk_local};
    }

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
        q,
        k,
        v,
        indices,
        resolved_scale,
        block_size,
        kernel_config,
        cache_batch_idx,
        chunk_start_idx,
        cluster_axis,
        block_cyclic);
}

}  // namespace ttnn::transformer
