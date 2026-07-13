// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/sparse_sdpa.hpp"
#include "ttnn/operations/transformer/sdpa/device/sparse_sdpa_device_operation.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <cmath>

namespace ttnn::transformer {

// Shared entry: resolves scale / block-cyclic / kernel-config and launches the prim op. Returns [O] or
// [O, m, l] (return_stats). The two public functions below differ only in return_stats + return shape.
static std::vector<ttnn::Tensor> sparse_sdpa_impl(
    const ttnn::Tensor& q,
    const ttnn::Tensor& kv,
    const ttnn::Tensor& indices,
    uint32_t v_dim,
    std::optional<float> scale,
    uint32_t k_chunk_size,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<uint32_t> cache_batch_idx,
    std::optional<uint32_t> block_cyclic_sp_axis,
    std::optional<uint32_t> block_cyclic_chunk_local,
    bool return_stats) {
    const uint32_t k_dim = q.logical_shape()[3];  // head dim, from the tensor
    const float resolved_scale = scale.value_or(1.0f / std::sqrt(static_cast<float>(k_dim)));

    // Block-cyclic remap: sp_axis and chunk_local must be provided together (or neither). `sp` is DERIVED from
    // the mesh (the cache's stripe count == the size of the SP mesh axis), so a caller cannot pass an sp that
    // disagrees with the device; `chunk_local` is then cross-checked against q's per-chip seq length.
    TT_FATAL(
        block_cyclic_sp_axis.has_value() == block_cyclic_chunk_local.has_value(),
        "sparse_sdpa: block_cyclic_sp_axis and block_cyclic_chunk_local must both be set or both unset");
    std::optional<ttnn::prim::BlockCyclicLayout> block_cyclic = std::nullopt;
    if (block_cyclic_sp_axis.has_value()) {
        const auto mesh_shape = q.device()->get_view().shape();
        const uint32_t sp_axis = block_cyclic_sp_axis.value();
        TT_FATAL(
            sp_axis < mesh_shape.dims(),
            "sparse_sdpa: block_cyclic_sp_axis ({}) out of range for mesh rank {}",
            sp_axis,
            mesh_shape.dims());
        const uint32_t sp = mesh_shape[sp_axis];
        const uint32_t tp = static_cast<uint32_t>(mesh_shape.mesh_size()) / sp;  // remaining (TP) device count
        const uint32_t chunk_local = block_cyclic_chunk_local.value();
        // chunk_local is one of exactly two values: q's per-chip seq length (no head->seq reshard) or tp*q_isl
        // (reshard slices q's seq across TP, leaving the cache's chunk_local unchanged). Anything else is a bug.
        const uint32_t q_isl = q.logical_shape()[2];
        TT_FATAL(
            chunk_local == q_isl || chunk_local == q_isl * tp,
            "sparse_sdpa: block_cyclic_chunk_local ({}) must be q_isl ({}) or tp*q_isl ({})",
            chunk_local,
            q_isl,
            q_isl * tp);
        block_cyclic = ttnn::prim::BlockCyclicLayout{sp, chunk_local};
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
        q,
        kv,
        indices,
        resolved_scale,
        v_dim,
        k_chunk_size,
        kernel_config,
        cache_batch_idx,
        block_cyclic,
        return_stats);
}

ttnn::Tensor sparse_sdpa(
    const ttnn::Tensor& q,
    const ttnn::Tensor& kv,
    const ttnn::Tensor& indices,
    uint32_t v_dim,
    std::optional<float> scale,
    uint32_t k_chunk_size,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<uint32_t> cache_batch_idx,
    std::optional<uint32_t> block_cyclic_sp_axis,
    std::optional<uint32_t> block_cyclic_chunk_local) {
    return sparse_sdpa_impl(
        q,
        kv,
        indices,
        v_dim,
        scale,
        k_chunk_size,
        compute_kernel_config,
        cache_batch_idx,
        block_cyclic_sp_axis,
        block_cyclic_chunk_local,
        /*return_stats=*/false)[0];
}

std::vector<ttnn::Tensor> sparse_sdpa_stats(
    const ttnn::Tensor& q,
    const ttnn::Tensor& kv,
    const ttnn::Tensor& indices,
    uint32_t v_dim,
    std::optional<float> scale,
    uint32_t k_chunk_size,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<uint32_t> cache_batch_idx,
    std::optional<uint32_t> block_cyclic_sp_axis,
    std::optional<uint32_t> block_cyclic_chunk_local) {
    return sparse_sdpa_impl(
        q,
        kv,
        indices,
        v_dim,
        scale,
        k_chunk_size,
        compute_kernel_config,
        cache_batch_idx,
        block_cyclic_sp_axis,
        block_cyclic_chunk_local,
        /*return_stats=*/true);
}

std::vector<ttnn::Tensor> sparse_sdpa_stats_shard_local(
    const ttnn::Tensor& q,
    const ttnn::Tensor& kv,
    const ttnn::Tensor& indices,
    const ttnn::Tensor& shard_id,
    uint32_t v_dim,
    uint32_t sp,
    uint32_t chunk_local,
    std::optional<float> scale,
    uint32_t k_chunk_size,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    // qr-ring building block: kv is ONLY this rank's stationary block-cyclic stripe [1,1,T/sp,K_DIM]. Which
    // stripe THIS device holds is read PER DEVICE from `shard_id` ([1,1,1,1] uint32, SP-sharded so device s
    // holds s), so ONE broadcast program is correct on 1 chip or a full mesh (the reader's shard_id accessor
    // resolves each device's value). sp/chunk_local describe the block-cyclic layout. Always returns stats
    // {O, m, l} (the cross-shard merge needs m,l). Empty rows yield the identity partial (see kernels).
    const uint32_t k_dim = q.logical_shape()[3];
    const float resolved_scale = scale.value_or(1.0f / std::sqrt(static_cast<float>(k_dim)));
    const bool any_fp8 = (kv.dtype() == ttnn::DataType::FP8_E4M3) || (q.dtype() == ttnn::DataType::FP8_E4M3);
    auto kernel_config = init_device_compute_kernel_config(
        tt::tt_metal::hal::get_arch(),
        compute_kernel_config,
        /*default_fidelity=*/MathFidelity::HiFi2,
        /*default_approx_mode=*/true,
        /*default_fp32_acc=*/any_fp8,
        /*default_l1_acc=*/false);
    ttnn::prim::BlockCyclicLayout bc{sp, chunk_local, /*shard_local=*/true};
    return ttnn::prim::sparse_sdpa(
        q,
        kv,
        indices,
        resolved_scale,
        v_dim,
        k_chunk_size,
        kernel_config,
        /*cache_batch_idx=*/std::nullopt,
        bc,
        /*return_stats=*/true,
        /*shard_id=*/shard_id);
}

}  // namespace ttnn::transformer
