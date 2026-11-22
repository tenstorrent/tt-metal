// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/sparse_sdpa_msa.hpp"
#include "ttnn/operations/transformer/sdpa/sparse_sdpa.hpp"
#include "ttnn/operations/transformer/sdpa/sdpa.hpp"
#include "ttnn/operations/transformer/sdpa/device/sparse_sdpa_msa_device_operation.hpp"
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
    std::optional<uint32_t> block_cyclic_chunk_local,
    bool enable_kv_block_cache,
    std::optional<SDPAPrecision> precision) {
    detail::warn_ignored_compute_kernel_config(compute_kernel_config);
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
        // The causal query positions are derived from the rank along cluster_axis against the cache's block-cyclic
        // SP striping, so the two must name the same mesh axis.
        TT_FATAL(
            !chunk_start_idx.has_value() || !cluster_axis.has_value() || cluster_axis.value() == sp_axis,
            "sparse_sdpa_msa: causal cluster_axis ({}) must equal block_cyclic_sp_axis ({})",
            cluster_axis.value_or(0),
            sp_axis);
    }

    // fp8 Q tilizes through the 32-bit DEST, so it defaults to an FP32-state recipe.
    const bool q_is_fp8 = (q.dtype() == ttnn::DataType::FP8_E4M3);
    const SDPAPrecision resolved_precision =
        precision.value_or(q_is_fp8 ? kSparseFp8Precision : SDPAPrecision::STANDARD);

    return ttnn::prim::sparse_sdpa_msa(
        q,
        k,
        v,
        indices,
        resolved_scale,
        block_size,
        resolved_precision,
        cache_batch_idx,
        chunk_start_idx,
        cluster_axis,
        block_cyclic,
        enable_kv_block_cache);
}

}  // namespace ttnn::transformer
