// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "indexer_score_device_operation_types.hpp"
#include "indexer_score_program_factory.hpp"
#include "ring_indexer_score_dsa_program_factory.hpp"

namespace ttnn::operations::experimental::indexer_score {

struct IndexerScoreDeviceOperation {
    using operation_attributes_t = indexer_score::operation_attributes_t;
    using tensor_args_t = indexer_score::tensor_args_t;
    using spec_return_value_t = indexer_score::spec_return_value_t;
    using tensor_return_value_t = indexer_score::tensor_return_value_t;
    // Classic factory for the unfused path (caller pre-gathers K); descriptor-based fused factory when the
    // op subsumes the SP all-gather (attrs.fused_ring set). select_program_factory picks by has_fused_ring().
    using program_factory_t =
        std::variant<program::IndexerScoreProgramFactory, program::RingIndexerScoreDsaMeshWorkloadFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    // Custom hash: runtime values (cache_batch_idx / kv_len / chunk_start_idx) are excluded so they reuse
    // one program; cluster_axis IS hashed.
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    // Re-checks the runtime values that can change on a hit (slot < B, kv_len bounds, chunk_start window).
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    // Matmul-FLOP performance model: reports ideal matmul cycles so tracy's util matches the math-util
    // test's mm_flops/(cores x cycles x peak). FLOPs count causal-valid tiles only.
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& q,
        const Tensor& k,
        const Tensor& weights,
        uint32_t chunk_start_idx,
        bool apply_relu,
        uint32_t num_groups,
        uint32_t block_size,
        bool synthesize_gate,
        float gate_scale,
        const IndexerScoreProgramConfig& program_config,
        const DeviceComputeKernelConfig& compute_kernel_config,
        std::optional<uint32_t> cache_batch_idx,
        std::optional<uint32_t> kv_len,
        std::optional<uint32_t> cluster_axis,
        std::optional<uint32_t> seq_subshard_axis,
        std::optional<BlockCyclicLayout> block_cyclic);
};

}  // namespace ttnn::operations::experimental::indexer_score

namespace ttnn::experimental {

// Two public frontends over one shared device op: the lightning indexer's two flavours differ only in
// fixed knobs, so each gets its own callable. Both share the program factory + 3 kernels (flavour = compile-
// time args) and produce a row-major bf16 score. Causality: key t visible to query s iff t <= chunk_start + s.
//
// BLOCK-CYCLIC K LAYOUT: the gathered K cache is a per-SP-shard slab (chunked prefill + SP all-gather), so the
// reader reads it back in natural token order via an invP remap. The interface matches
// ttnn.transformer.sparse_sdpa: the caller names the MESH AXIS the cache was striped over
// (block_cyclic_sp_axis) and passes the per-shard chunk length (block_cyclic_chunk_local); `sp` is DERIVED
// from the mesh shape on that axis, so a caller cannot pass an sp that disagrees with the device.
// block_cyclic_chunk_local must be q_isl (seq sharded only on the SP axis) or tp*q_isl (tp = mesh/sp). Both
// set together, or neither = contiguous K (no remap); sp==1 is the identity. Seq sharded across BOTH axes
// (chunk_local == tp*q_isl, tp>1) is allowed ONLY with cluster_axis=None (flat row-major linearization over
// all devices == a row-major nested 2D seq shard); with a NAMED cluster_axis it is rejected (chunk_start
// would miss the second axis's seq offset).

// DeepSeek-V3.2 DSA / GLM-5 (ttnn.experimental.indexer_score_dsa):
//   score[b, 0, s, t] = sum_h relu(q[b,h,s,:] . k[b,t,:]) * weights[b,h,s]
// q [B,Hi,Sq,D], k [B,1,T,D], weights [B,Hi,Sq,1] -> score [B,1,Sq,T] (all heads relu'd + summed).
// cache_batch_idx/kv_len/chunk_start_idx are re-applied each dispatch and hash-excluded (no recompile); see
// the nanobind docs for their full semantics. OMIT chunk_start_idx on a mesh (deduced as T - sp_ring*Sq).
ttnn::Tensor indexer_score_dsa(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& weights,
    std::optional<uint32_t> chunk_start_idx = std::nullopt,
    const ttnn::operations::experimental::indexer_score::IndexerScoreProgramConfig& program_config = {},
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    std::optional<uint32_t> cache_batch_idx = std::nullopt,
    std::optional<uint32_t> kv_len = std::nullopt,
    std::optional<uint32_t> cluster_axis = std::nullopt,
    std::optional<uint32_t> seq_subshard_axis = std::nullopt,
    std::optional<uint32_t> block_cyclic_sp_axis = std::nullopt,
    std::optional<uint32_t> block_cyclic_chunk_local = std::nullopt);

// MiniMax-M3 MSA (ttnn.experimental.indexer_score_msa):
//   score[b, g, s, t] = sum_{h in group g} (q[b,h,s,:] . k[b,t,:]) * scale
// Raw dot, no learned gates -- only a 1/sqrt(d) `scale` (synthesized as a constant gate, so no weights
// tensor). q [B,Hi,Sq,D], k [B,1,T,D] -> score [B,num_groups,Sq,T_out].
// num_groups: G output planes (no cross-group sum); G==1 = TP=4 group-aligned (Hi=1/device), G>1 needs all
//   heads resident + k_chunk>=64. block_size: 0 = full [B,G,Sq,T]; >0 = block-max-pool -> [B,G,Sq,T/bs].
// chunk_start_idx / cluster_axis / cache_batch_idx / kv_len: same semantics as indexer_score_dsa (the last
// two are runtime, hash-excluded pass-throughs -- no recompile when the slot or valid length changes).
// num_groups is required (no default): per-GQA-group selection is MSA's purpose, so the caller must state
// the group count explicitly. It is placed before the defaulted optionals so the signature stays well-formed.
ttnn::Tensor indexer_score_msa(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    uint32_t num_groups,
    std::optional<uint32_t> chunk_start_idx = std::nullopt,
    float scale = 1.0f,
    uint32_t block_size = 0,
    const ttnn::operations::experimental::indexer_score::IndexerScoreProgramConfig& program_config = {},
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    std::optional<uint32_t> cache_batch_idx = std::nullopt,
    std::optional<uint32_t> kv_len = std::nullopt,
    std::optional<uint32_t> cluster_axis = std::nullopt,
    std::optional<uint32_t> block_cyclic_sp_axis = std::nullopt,
    std::optional<uint32_t> block_cyclic_chunk_local = std::nullopt);

// FUSED DSA (ttnn.experimental.ring_indexer_score_dsa): subsumes the SP all-gather. Instead of pre-gathering
// K, the caller hands this chip's LOCAL K shard `k_local` [B,1,sll,D] (the all-gather input) plus a
// pre-allocated `k` [B,1,T,D] persistent buffer (the gather output, seeded with the local slab). One program
// co-schedules the ring_attention all-gather + the indexer compute, overlapping fabric transport with scoring:
// the reader gates each K band on only the SP shards its tiles land in (per-band overlap, not a coarse whole-
// gather barrier), then scores exactly as indexer_score_dsa. Same score semantics +
// block-cyclic remap as indexer_score_dsa. Ring runs over `cluster_axis` (the SP mesh axis) on `topology`
// (Linear on a non-torus grid). `ag_multi_device_global_semaphore` are the all-gather's own out-ready
// semaphores; `ag_sub_device_id` scopes the AG worker cores (kept disjoint from the compute rectangle).
ttnn::Tensor ring_indexer_score_dsa(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& weights,
    const ttnn::Tensor& k_local,
    const std::vector<tt::tt_metal::GlobalSemaphore>& ag_multi_device_global_semaphore,
    uint32_t cluster_axis,
    ttnn::ccl::Topology topology,
    uint32_t num_links = 1,
    std::optional<tt::tt_metal::SubDeviceId> ag_sub_device_id = std::nullopt,
    std::optional<uint32_t> chunk_start_idx = std::nullopt,
    const ttnn::operations::experimental::indexer_score::IndexerScoreProgramConfig& program_config = {},
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    std::optional<uint32_t> cache_batch_idx = std::nullopt,
    std::optional<uint32_t> kv_len = std::nullopt,
    std::optional<uint32_t> block_cyclic_sp_axis = std::nullopt,
    std::optional<uint32_t> block_cyclic_chunk_local = std::nullopt);

}  // namespace ttnn::experimental
