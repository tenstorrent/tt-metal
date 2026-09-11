// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/vsa_ring_sdpa_device_operation.hpp"

#include <bit>
#include <utility>

#include <tt-metalium/hal.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/vsa_sdpa_device_operation.hpp"

namespace ttnn::prim {

namespace {

// The plain vsa_sdpa validators see the GLOBAL sequence: swap the local K/V shard for the gathered buffers.
VsaSdpaInputs as_gathered_inputs(const VsaRingSdpaInputs& t) {
    VsaSdpaInputs g = t.vsa;
    g.k = t.gathered_k;
    g.v = t.gathered_v;
    return g;
}

void validate_ring(const VsaRingSdpaParams& attrs, const VsaRingSdpaInputs& t) {
    const auto& k = t.vsa.k;
    const auto& v = t.vsa.v;
    const auto& gk = t.gathered_k;
    const auto& gv = t.gathered_v;
    TT_FATAL(
        attrs.vsa.streaming && !attrs.vsa.distributed && !t.vsa.stream_order.has_value(),
        "vsa_ring_sdpa: streaming kernel only, no distributed mode, no stream_order");
    TT_FATAL(
        k.logical_shape() == v.logical_shape(),
        "vsa_ring_sdpa: local k/v shapes differ ({} vs {})",
        k.logical_shape(),
        v.logical_shape());
    TT_FATAL(gk.logical_shape() == gv.logical_shape(), "vsa_ring_sdpa: gathered k/v shapes differ");
    TT_FATAL(
        k.dtype() == gk.dtype() && v.dtype() == gv.dtype(), "vsa_ring_sdpa: local and gathered K/V dtypes must match");
    for (const Tensor* tp : {&k, &v}) {
        TT_FATAL(tp->layout() == Layout::TILE, "vsa_ring_sdpa: local k/v must be TILE");
        TT_FATAL(
            tp->memory_config().buffer_type() == BufferType::DRAM && !tp->memory_config().is_sharded(),
            "vsa_ring_sdpa: local k/v must be interleaved DRAM");
    }
    const auto ks = k.logical_shape();
    const auto gs = gk.logical_shape();
    TT_FATAL(ks.rank() == 4 && gs.rank() == 4, "vsa_ring_sdpa: k and gathered k must be rank 4");
    TT_FATAL(
        ks[0] == gs[0] && ks[1] == gs[1] && ks[3] == gs[3],
        "vsa_ring_sdpa: k {} and gathered k {} differ outside the sequence dim",
        ks,
        gs);
    TT_FATAL(attrs.ag.ring_size >= 2, "vsa_ring_sdpa: ring_size must be >= 2 (got {})", attrs.ag.ring_size);
    TT_FATAL(
        ks[2] * attrs.ag.ring_size == gs[2],
        "vsa_ring_sdpa: gathered sequence ({}) must be ring_size ({}) x local sequence ({})",
        gs[2],
        attrs.ag.ring_size,
        ks[2]);
    TT_FATAL(
        attrs.vsa.block_size > 0 && ks[2] % attrs.vsa.block_size == 0,
        "vsa_ring_sdpa: local sequence ({}) must be a multiple of block_size ({})",
        ks[2],
        attrs.vsa.block_size);
    TT_FATAL(attrs.ag.num_links >= 1, "vsa_ring_sdpa: num_links must be >= 1");
    TT_FATAL(attrs.ag.topology == ttnn::ccl::Topology::Ring, "vsa_ring_sdpa: topology must be Ring");
    TT_FATAL(attrs.ag.semaphore.size() >= 2, "vsa_ring_sdpa: two GlobalSemaphores [backward, forward] are required");
    TT_FATAL(attrs.ag.cluster_axis.has_value(), "vsa_ring_sdpa: cluster_axis is required");
}

}  // namespace

void VsaRingSdpaOperation::validate_on_program_cache_miss(const VsaRingSdpaParams& attrs, const VsaRingSdpaInputs& t) {
    validate_ring(attrs, t);
    VsaSdpaOperation::validate_on_program_cache_miss(attrs.vsa, as_gathered_inputs(t));
}

void VsaRingSdpaOperation::validate_on_program_cache_hit(const VsaRingSdpaParams& attrs, const VsaRingSdpaInputs& t) {
    validate_ring(attrs, t);
    VsaSdpaOperation::validate_on_program_cache_hit(attrs.vsa, as_gathered_inputs(t));
}

VsaRingSdpaOperation::spec_return_value_t VsaRingSdpaOperation::compute_output_specs(
    const VsaRingSdpaParams& attrs, const VsaRingSdpaInputs& t) {
    return VsaSdpaOperation::compute_output_specs(attrs.vsa, t.vsa);
}

VsaRingSdpaOperation::tensor_return_value_t VsaRingSdpaOperation::create_output_tensors(
    const VsaRingSdpaParams& attrs, const VsaRingSdpaInputs& t) {
    return create_device_tensor(compute_output_specs(attrs, t), t.vsa.q.device());
}

ttsl::hash::hash_t VsaRingSdpaOperation::compute_program_hash(
    const VsaRingSdpaParams& attrs, const VsaRingSdpaInputs& t) {
    // Every shape is hashed (the kernels bake strides and widths as compile-time args). The all-gather
    // attributes hash their structural fields only: the GlobalSemaphores are excluded and re-applied per
    // dispatch by the mesh workload factory.
    const auto& a = attrs.vsa;
    return tt::tt_metal::operation::hash_operation<VsaRingSdpaOperation>(
        std::bit_cast<uint32_t>(a.scale),
        a.block_size,
        a.list_len,
        a.exempt_ids,
        a.coarse_slots_shift,
        a.coarse_real_per_shard,
        a.dense_row_hint,
        t.vsa.dense_row_mask.has_value(),
        a.compute_kernel_config,
        attrs.ag,
        attrs.ccl_core_grid_offset,
        t.vsa.q.logical_shape(),
        t.vsa.q.dtype(),
        t.vsa.k.logical_shape(),
        t.vsa.k.dtype(),
        t.vsa.v.dtype(),
        t.gathered_k.logical_shape(),
        t.gathered_k.dtype(),
        t.vsa.indices.logical_shape());
}

Tensor vsa_ring_sdpa(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& indices,
    const Tensor& block_counts,
    const Tensor& persistent_output_buffer_k,
    const Tensor& persistent_output_buffer_v,
    float scale,
    uint32_t block_size,
    ttnn::DeviceComputeKernelConfig compute_kernel_config,
    uint32_t list_len,
    std::vector<uint32_t> exempt_ids,
    std::optional<Tensor> dense_row_mask,
    uint32_t coarse_slots_shift,
    uint32_t coarse_real_per_shard,
    std::vector<uint32_t> dense_row_hint,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    uint32_t num_links,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    tt::tt_metal::CoreCoord ccl_core_grid_offset,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id) {
    using OperationType = VsaRingSdpaOperation;

    const auto& mesh_view = mesh_device.get_view();
    TT_FATAL(mesh_view.is_mesh_2d(), "vsa_ring_sdpa: the cluster_axis API needs a 2D mesh");
    const std::size_t ring_size = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    VsaSdpaParams vsa{
        .scale = scale,
        .block_size = block_size,
        .k_chunk_blocks = 1,
        .streaming = true,
        .list_len = list_len,
        .exempt_ids = std::move(exempt_ids),
        .coarse_slots_shift = coarse_slots_shift,
        .coarse_real_per_shard = coarse_real_per_shard,
        .distributed = false,
        .dense_row_hint = std::move(dense_row_hint),
        .compute_kernel_config = compute_kernel_config,
    };
    ttnn::experimental::prim::RingAttentionAllGatherAsyncParams ag{
        {},
        /*dim=*/2,
        num_links,
        static_cast<uint32_t>(ring_size),
        persistent_output_buffer_k.memory_config(),
        topology,
        multi_device_global_semaphore,
        subdevice_id,
        cluster_axis,
        ttnn::ccl::CoreAllocationStrategy::ROW_MAJOR};

    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .vsa = std::move(vsa), .ag = std::move(ag), .ccl_core_grid_offset = ccl_core_grid_offset},
        OperationType::tensor_args_t{
            .vsa =
                VsaSdpaInputs{
                    .q = q,
                    .k = k,
                    .v = v,
                    .indices = indices,
                    .block_counts = block_counts,
                    .dense_row_mask = std::move(dense_row_mask),
                    .stream_order = std::nullopt,
                },
            .gathered_k = persistent_output_buffer_k,
            .gathered_v = persistent_output_buffer_v});
}

}  // namespace ttnn::prim
