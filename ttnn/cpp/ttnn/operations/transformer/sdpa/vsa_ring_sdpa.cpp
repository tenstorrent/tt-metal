// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/vsa_ring_sdpa.hpp"

#include <cmath>
#include <utility>

#include <tt-metalium/hal.hpp>
#include "ttnn/operations/transformer/sdpa/device/vsa_ring_sdpa_device_operation.hpp"

namespace ttnn::transformer {

ttnn::Tensor vsa_ring_sdpa(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& indices,
    const ttnn::Tensor& block_counts,
    const ttnn::Tensor& persistent_output_buffer_k,
    const ttnn::Tensor& persistent_output_buffer_v,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    uint32_t num_links,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    const std::string& gather,
    uint32_t num_workers_per_link,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    std::optional<float> scale,
    uint32_t block_size,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    uint32_t list_len,
    std::vector<uint32_t> exempt_ids,
    std::optional<ttnn::Tensor> dense_row_mask,
    uint32_t coarse_slots_shift,
    uint32_t coarse_real_per_shard,
    std::vector<uint32_t> dense_row_hint) {
    ttnn::prim::VsaRingGather gather_mode;
    if (gather == "ring_attention") {
        gather_mode = ttnn::prim::VsaRingGather::RingAttention;
    } else if (gather == "fused_kv") {
        gather_mode = ttnn::prim::VsaRingGather::FusedKv;
    } else {
        TT_THROW("vsa_ring_sdpa: gather must be \"ring_attention\" or \"fused_kv\" (got \"{}\")", gather);
    }
    const uint32_t d = q.logical_shape()[3];
    const float resolved_scale = scale.value_or(1.0f / std::sqrt(static_cast<float>(d)));
    // Same numerics contract as vsa_sdpa: HiFi2, exact exp (lossless mandate), bf16 accumulation.
    auto kernel_config = init_device_compute_kernel_config(
        tt::tt_metal::hal::get_arch(),
        compute_kernel_config,
        /*default_fidelity=*/MathFidelity::HiFi2,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/false,
        /*default_l1_acc=*/false);
    return ttnn::prim::vsa_ring_sdpa(
        q,
        k,
        v,
        indices,
        block_counts,
        persistent_output_buffer_k,
        persistent_output_buffer_v,
        resolved_scale,
        block_size,
        kernel_config,
        list_len,
        std::move(exempt_ids),
        std::move(dense_row_mask),
        coarse_slots_shift,
        coarse_real_per_shard,
        std::move(dense_row_hint),
        multi_device_global_semaphore,
        num_links,
        cluster_axis,
        mesh_device,
        topology,
        gather_mode,
        num_workers_per_link,
        subdevice_id);
}

}  // namespace ttnn::transformer
