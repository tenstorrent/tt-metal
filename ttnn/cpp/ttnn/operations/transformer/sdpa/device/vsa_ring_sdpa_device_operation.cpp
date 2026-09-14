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

void validate_ring(const VsaRingSdpaParams& attrs, const VsaRingSdpaInputs& t) {
    const auto& q = t.vsa.q;
    TT_FATAL(
        attrs.vsa.streaming && !attrs.vsa.distributed && !t.vsa.stream_order.has_value(),
        "vsa_ring_sdpa: streaming kernel only, no distributed mode, no stream_order");
    for (const Tensor* tp : {&t.vsa.k, &t.vsa.v, &t.gathered_k, &t.gathered_v}) {
        TT_FATAL(tp->layout() == Layout::TILE, "vsa_ring_sdpa: K/V tensors must be TILE");
        TT_FATAL(
            tp->memory_config().buffer_type() == BufferType::DRAM && !tp->memory_config().is_sharded(),
            "vsa_ring_sdpa: K/V tensors must be interleaved DRAM");
        TT_FATAL(tp->padded_shape() == tp->logical_shape(), "vsa_ring_sdpa: K/V tensors must not be padded");
        TT_FATAL(tp->dtype() == t.vsa.k.dtype(), "vsa_ring_sdpa: K/V and gathered K/V dtypes must match");
    }
    const auto qs = q.logical_shape();
    const auto ks = t.vsa.k.logical_shape();
    const auto gs = t.gathered_k.logical_shape();
    TT_FATAL(
        qs.rank() == 4 && ks.rank() == 4 && gs.rank() == 4, "vsa_ring_sdpa: q, k/v and gathered k/v must be rank 4");
    TT_FATAL(
        ks == t.vsa.v.logical_shape() && gs == t.gathered_v.logical_shape(),
        "vsa_ring_sdpa: k and v ({} / {}) and their gathered buffers ({} / {}) must match",
        ks,
        t.vsa.v.logical_shape(),
        gs,
        t.gathered_v.logical_shape());
    const uint32_t H = qs[1];
    const uint32_t d = qs[3];
    TT_FATAL(
        ks[0] == 1 && ks[1] == H && ks[3] == d,
        "vsa_ring_sdpa: k/v must be [1, H, T_local, d] (got {} for q {})",
        ks,
        qs);
    TT_FATAL(
        gs[0] == 1 && gs[1] == H && gs[3] == d,
        "vsa_ring_sdpa: gathered k/v must be [1, H, T, d] (got {} for q {})",
        gs,
        qs);
    TT_FATAL(attrs.ag.ring_size >= 2, "vsa_ring_sdpa: ring_size must be >= 2 (got {})", attrs.ag.ring_size);
    TT_FATAL(
        ks[2] * attrs.ag.ring_size == gs[2],
        "vsa_ring_sdpa: gathered sequence ({}) must be ring_size ({}) x local sequence ({})",
        gs[2],
        attrs.ag.ring_size,
        ks[2]);
    TT_FATAL(
        attrs.vsa.block_size > 0 && ks[2] % attrs.vsa.block_size == 0 && ks[2] % 32 == 0,
        "vsa_ring_sdpa: local sequence ({}) must be a multiple of block_size ({}) and of 32",
        ks[2],
        attrs.vsa.block_size);
    TT_FATAL(
        attrs.ag.num_links >= 1 && attrs.num_workers_per_link >= 1,
        "vsa_ring_sdpa: num_links and num_workers_per_link must be >= 1");
    TT_FATAL(attrs.ag.topology == ttnn::ccl::Topology::Ring, "vsa_ring_sdpa: topology must be Ring");
    TT_FATAL(attrs.ag.semaphore.size() >= 2, "vsa_ring_sdpa: two GlobalSemaphores are required");
    // The gather's sender cores (per link: two directions of workers + MUX) fill the compute grid from (0, 0) in
    // row-major order. Configurations that spill into a second grid row hang on the 4x8 Blackhole galaxy (2 links x
    // 3+ workers; 1 link x 4 workers and 2 links x 2 workers run) -- root cause open, so refuse them here rather
    // than time out on the device.
    {
        const auto grid = q.device()->compute_with_storage_grid_size();
        const uint32_t mux = attrs.num_workers_per_link == 1 ? 0u : 1u;
        const uint32_t senders = attrs.ag.num_links * 2 * (attrs.num_workers_per_link + mux);
        TT_FATAL(
            senders <= grid.x,
            "vsa_ring_sdpa: {} sender cores ({} links x 2 directions x ({} workers + {} mux)) exceed one grid row of "
            "{} "
            "cores; senders spilling into a second row hang (open issue). Reduce num_workers_per_link or num_links.",
            senders,
            attrs.ag.num_links,
            attrs.num_workers_per_link,
            mux,
            grid.x);
    }
    TT_FATAL(attrs.ag.cluster_axis.has_value(), "vsa_ring_sdpa: cluster_axis is required");
}

// The plain vsa_sdpa validation needs k/v with the global sequence: check what it would check
// (indices/counts against T = gathered length) with the ring's own shape facts.
void validate_vsa_contract(const VsaRingSdpaParams& attrs, const VsaRingSdpaInputs& t) {
    const auto& q = t.vsa.q;
    const auto& idx = t.vsa.indices;
    const auto& counts = t.vsa.block_counts;
    const uint32_t H = q.logical_shape()[1];
    const uint32_t S = q.logical_shape()[2];
    const uint32_t T = t.gathered_k.logical_shape()[2];
    TT_FATAL(S > 0 && S % 64 == 0, "vsa_ring_sdpa: q sequence length ({}) must be a positive multiple of 64", S);
    const uint32_t n_kv_blocks = T / attrs.vsa.block_size;
    const uint32_t n_q_tiles = S / 64;
    const auto is = idx.logical_shape();
    TT_FATAL(
        is.rank() == 4 && is[0] == 1 && is[1] == H && is[2] == n_q_tiles,
        "vsa_ring_sdpa: indices must be [1,H,S/64,W] (got {} for H {}, S/64 {})",
        is,
        H,
        n_q_tiles);
    TT_FATAL(
        idx.layout() == Layout::ROW_MAJOR && counts.layout() == Layout::ROW_MAJOR,
        "vsa_ring_sdpa: indices/block_counts must be ROW_MAJOR");
    TT_FATAL(
        attrs.vsa.list_len <= is[3],
        "vsa_ring_sdpa: list_len ({}) exceeds the indices width ({})",
        attrs.vsa.list_len,
        is[3]);
    TT_FATAL(attrs.vsa.exempt_ids.size() <= 32, "vsa_ring_sdpa: at most 32 exempt block ids");
    for (uint32_t b : attrs.vsa.exempt_ids) {
        TT_FATAL(b < n_kv_blocks, "vsa_ring_sdpa: exempt block id {} out of range ({} blocks)", b, n_kv_blocks);
    }
    for (uint32_t r : attrs.vsa.dense_row_hint) {
        TT_FATAL(r < n_q_tiles, "vsa_ring_sdpa: dense_row_hint row {} out of range (S/64 = {})", r, n_q_tiles);
    }
    const auto cs = counts.logical_shape();
    TT_FATAL(
        cs.rank() == 4 && cs[0] == 1 && cs[1] == 1 && cs[2] == 1 && cs[3] >= n_kv_blocks,
        "vsa_ring_sdpa: block_counts must be [1,1,1,Wc] with Wc >= T/block_size (got {} for {} blocks)",
        cs,
        n_kv_blocks);
    if (attrs.vsa.coarse_slots_shift != 0) {
        TT_FATAL(
            attrs.vsa.coarse_slots_shift < 16 && attrs.vsa.coarse_real_per_shard > 0 &&
                attrs.vsa.coarse_real_per_shard <= (1u << attrs.vsa.coarse_slots_shift),
            "vsa_ring_sdpa: bad coarse numbering: shift {} real/shard {}",
            attrs.vsa.coarse_slots_shift,
            attrs.vsa.coarse_real_per_shard);
    }
}

}  // namespace

void VsaRingSdpaOperation::validate_on_program_cache_miss(const VsaRingSdpaParams& attrs, const VsaRingSdpaInputs& t) {
    TT_FATAL(tt::tt_metal::hal::get_arch() == tt::ARCH::BLACKHOLE, "vsa_ring_sdpa is Blackhole-only");
    TT_FATAL(t.vsa.q.dtype() == DataType::BFLOAT16, "vsa_ring_sdpa: q must be bf16");
    TT_FATAL(t.vsa.k.dtype() == DataType::BFLOAT16, "vsa_ring_sdpa: k/v must be bf16");
    TT_FATAL(
        t.vsa.indices.dtype() == DataType::UINT32 && t.vsa.block_counts.dtype() == DataType::UINT32,
        "vsa_ring_sdpa: indices/block_counts must be uint32");
    validate_ring(attrs, t);
    validate_vsa_contract(attrs, t);
}

void VsaRingSdpaOperation::validate_on_program_cache_hit(const VsaRingSdpaParams& attrs, const VsaRingSdpaInputs& t) {
    validate_ring(attrs, t);
    validate_vsa_contract(attrs, t);
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
        attrs.num_workers_per_link,
        t.vsa.q.logical_shape(),
        t.vsa.q.dtype(),
        t.vsa.k.logical_shape(),
        t.vsa.k.dtype(),
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
    uint32_t num_workers_per_link,
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
            .vsa = std::move(vsa), .ag = std::move(ag), .num_workers_per_link = num_workers_per_link},
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
