// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_fabric2d.hpp"
#include "device/combine_fabric2d_device_operation.hpp"
#include "device/kernels/dataflow/combine_fabric2d_kernel_interface.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

ttnn::Tensor combine_fabric2d(
    const ttnn::Tensor& dispatched_buffer,
    const ttnn::Tensor& dispatched_metadata,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& expert_region_offsets,
    const ttnn::Tensor& expert_offsets,
    uint32_t experts_per_chip,
    uint32_t num_experts_per_tok,
    uint32_t seq_len_per_chip,
    uint32_t cluster_axis,
    uint32_t num_links,
    uint32_t fwd_vc_pages,
    tt::tt_fabric::Topology topology,
    const tt::tt_metal::MemoryConfig& memory_config,
    const std::optional<ttnn::Tensor>& forwarding_buffer) {
    return ttnn::prim::combine_fabric2d(
        dispatched_buffer.device(),
        dispatched_buffer,
        dispatched_metadata,
        expert_token_counts,
        expert_region_offsets,
        expert_offsets,
        experts_per_chip,
        num_experts_per_tok,
        seq_len_per_chip,
        cluster_axis,
        num_links,
        fwd_vc_pages,
        topology,
        memory_config,
        forwarding_buffer);
}

ttnn::Tensor combine_fabric2d_forwarding_buffer(
    ttnn::MeshDevice* mesh_device, uint32_t emb_dim, uint32_t fwd_vc_pages, uint32_t cluster_axis, uint32_t num_links) {
    // The op's own checks on these, which the spec relies on and would otherwise not run until the first call.
    TT_FATAL(
        cluster_axis < mesh_device->shape().dims(),
        "combine_fabric2d_forwarding_buffer: cluster_axis {} is out of range for a {} mesh",
        cluster_axis,
        mesh_device->shape());
    TT_FATAL(
        fwd_vc_pages >= cmbf2d::MIN_VC_PAGES,
        "combine_fabric2d_forwarding_buffer: fwd_vc_pages must be at least {} (got {})",
        cmbf2d::MIN_VC_PAGES,
        fwd_vc_pages);
    // Only the fields the spec reads.
    const CombineFabric2dParams params{
        .device = mesh_device,
        .axis = cluster_axis,
        .num_links = num_links,
        .fwd_vc_pages = fwd_vc_pages,
    };
    return create_device_tensor(
        forwarding_buffer_spec(params, emb_dim * static_cast<uint32_t>(sizeof(uint16_t))), mesh_device);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
