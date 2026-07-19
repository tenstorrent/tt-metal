// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <optional>

#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"  // ttnn::ccl::Topology
#include <tt-metalium/base_types.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/global_semaphore.hpp>

namespace ttnn::operations::experimental::indexer_score {

// Work-unit knobs (elements, tile-aligned; SDPAProgramConfig analogue).
// One unit = q_chunk rows x k_chunk keys; heads stream in head_group blocks.
struct IndexerScoreProgramConfig {
    std::size_t q_chunk_size = 32;
    std::size_t k_chunk_size = 32;
    std::size_t head_group_size = 1;  // heads resident at once; 1 always fits L1, raise for perf (0 = all)
    // Benchmarking-only caps on the compute rectangle (0 = use the full device compute grid). The classic
    // (unfused) factory clamps grid_x/grid_y to these, so a standalone indexer can be pinned to the SAME core
    // budget the fused ring op leaves after reserving its AG column (e.g. cap grid_x to 10 on an 11x10 grid to
    // match the fused op's 100-core compute rectangle for an apples-to-apples AG-hiding measurement). Hashed via
    // reflection on this struct, so a capped and an uncapped program never collide. Ignored by the fused factory.
    std::size_t max_core_grid_x = 0;
    std::size_t max_core_grid_y = 0;
};

// Resolve head_group_size to a concrete head count (0 = all Hi). Single-sourced so validate and the
// factory can't drift on the "0 = all" contract.
inline uint32_t resolve_head_group(const IndexerScoreProgramConfig& cfg, uint32_t Hi) {
    return cfg.head_group_size == 0 ? Hi : static_cast<uint32_t>(cfg.head_group_size);
}

// Block-cyclic (per-SP-shard) K cache layout. Chunked prefill stores each of `sp` SP shards' per-chunk slab
// (chunk_local keys) back-to-back in its local cache, so after the SP all-gather the [B,1,T,D] k holds keys in
// a PERMUTED physical order: physical row r holds the natural token at
// P[r] = (lr/chunk_local)*chunk_global + c*chunk_local + (lr%chunk_local), where c = r/sll, lr = r%sll,
// sll = T/sp, chunk_global = sp*chunk_local. The reader reads K back in LOGICAL token order (invP per tile),
// so the causal mask and block-max-pool -- both keyed on the logical column -- stay byte-identical and the
// score columns come out in natural token order. RESOLVED at the ttnn entry (interface + naming match
// sparse_sdpa): the caller names the mesh axis the cache was striped over (block_cyclic_sp_axis) and passes
// the per-shard chunk length (block_cyclic_chunk_local); sp = the mesh extent on that axis (DERIVED, not
// free). Hashed (it shapes the reader binary). sp == 1 is the identity, represented as no block_cyclic at all.
struct BlockCyclicLayout {
    uint32_t sp;           // SP shard count the cache was gathered across (derived from the mesh sp axis)
    uint32_t chunk_local;  // per-shard chunk length (elements); == chunk_size_global / sp == per-chip seq_len
};

// Ring-fused indexer: the op subsumes the SP all-gather. Instead of the caller pre-gathering K, it hands the
// per-chip LOCAL K shard (tensor_args.k_local) as the all-gather INPUT and the full [B,1,T,D] persistent buffer
// (tensor_args.k) as the all-gather OUTPUT; the fused program factory co-schedules the ring_attention all-gather
// (the only Linear+fuse-capable AG) into the SAME program as the indexer compute, wiring a producer->consumer
// semaphore handshake so the reader starts scoring once the gather lands. Scalar AG config only (tensors live in
// tensor_args); NOT hashed (the fused path is selected by has_value(), and these carry no binary-shaping info the
// reader's FUSED_RING define + block-cyclic defines don't already cover). ring_size/ring_index are DERIVED from
// the mesh + coordinate (cluster_axis), not stored.
struct FusedRingConfig {
    uint32_t num_links{1};  // fabric links for the gather
    ttnn::ccl::Topology topology{ttnn::ccl::Topology::Linear};
    std::vector<tt::tt_metal::GlobalSemaphore> ag_semaphore;  // the all-gather's own out-ready semaphores
    std::optional<tt::tt_metal::SubDeviceId> ag_sub_device_id{std::nullopt};
    // NOTE: the all-gather concat dim is structurally fixed to seq (dim 2) -- the reader's block-cyclic
    // permutation assumes it -- so it is a named constant at the AG call site, not a configurable field. The AG
    // workers' grid offset is likewise computed by the factory (reserved-column math), not carried here.
};

struct operation_attributes_t {
    // Absolute chunk_start of rank 0. Rank r uses chunk_start_idx + r*Sq; the per-device value is derived
    // host-side and passed to compute as a RUNTIME arg (hash-excluded), so distinct values reuse one program.
    uint32_t chunk_start_idx{0};             // elements, tile-aligned
    std::optional<uint32_t> cluster_axis{};  // mesh axis that is the SP ring; unset = linear device order
    // Second mesh axis (TP) that the query sequence is ALSO block-cyclically sub-sharded over, on top of the
    // SP block-cyclic layout. When set (alongside a named cluster_axis + block_cyclic), each device owns a
    // Sq-row sub-range [tp_rank*Sq, (tp_rank+1)*Sq) of its SP chip's chunk_local-wide slab; the causal geometry
    // adds that TP sub-offset to the exact block-cyclic position. unset = query sharded on the SP axis only.
    std::optional<uint32_t> seq_subshard_axis{};
    // ReLU on each per-head q.kT before the gate-mul. true = DSA/GLM (relu(q.k)*w); false = raw dot (M3 MSA).
    // Compile-time, so the true path is byte-identical to before.
    bool apply_relu{true};
    // Output groups. 1 = sum ALL Hi heads -> [B,1,Sq,T] (DSA/GLM). G>1 = partition into G groups of Hi/G,
    // sum within each -> [B,G,Sq,T] (M3 per-GQA-group). Compile-time (G==1 byte-identical). G>1 needs all
    // heads resident (head_group_size 0 or Hi) and the full-strip path (k_chunk_size>=64).
    uint32_t num_groups{1};
    // Block-max-pool width in keys. 0 = no pooling -> [B,G,Sq,T]. >0 = max over each block -> [B,G,Sq,T/bs]
    // (M3 block selection). Compile-time (block_tiles = block_size/TILE_WIDTH; bs==0 byte-identical). >0 needs
    // bs % TILE_WIDTH == 0, T % bs == 0, k_chunk_size % bs == 0, and blocks-per-unit <= TILE_HEIGHT.
    uint32_t block_size{0};
    // MSA has no learned gates, only a constant 1/sqrt(d) scale: when true the reader fills cb_w with
    // gate_scale in L1 (no weights tensor, no extra fill op) instead of reading DRAM. The weights handle in
    // tensor_args is then an unused placeholder (the caller passes q). Compile-time + hashed (changes the
    // reader binary); gate_scale is hashed too so distinct scales get distinct programs. DSA: false (reads
    // its learned weights), byte-identical to before.
    bool synthesize_gate{false};
    float gate_scale{1.0f};
    IndexerScoreProgramConfig program_config{};
    // Resolved (not optional) so it is part of the program-cache key; the callable fills it from the user's
    // optional config, defaulting math_fidelity to the dtype-derived choice.
    DeviceComputeKernelConfig compute_kernel_config{};
    // Indexed KV cache: selects the batch slot of a shared [B,1,T,D] k (page ids offset by
    // cache_batch_idx * Tt * Dt). NOT hashed, re-applied each dispatch, so switching slots does NOT recompile.
    std::optional<uint32_t> cache_batch_idx{std::nullopt};
    bool has_indexed_kv_cache() const { return cache_batch_idx.has_value(); }
    // Runtime KV length: the valid prefix this dispatch (rest masked). NOT hashed, so growing kv_len <= T
    // reuses ONE program. grid/work-split/output width stay keyed on the hashed T. nullopt == T.
    std::optional<uint32_t> kv_len{std::nullopt};
    bool has_runtime_kv_len() const { return kv_len.has_value(); }
    // Resolved block-cyclic (per-SP-shard) K layout. When set, the reader remaps each logical k-tile to its
    // physical (permuted) tile, presenting K in natural token order. HASHED (sp/chunk_local shape the reader
    // binary via compile-time arguments). nullopt == contiguous K
    // (which is also what sp == 1 resolves to, since that is the identity permutation).
    std::optional<BlockCyclicLayout> block_cyclic{std::nullopt};
    bool has_block_cyclic() const { return block_cyclic.has_value(); }
    // Ring-fused all-gather config (see FusedRingConfig). nullopt = the classic unfused path (caller pre-gathers
    // K), which selects the classic program factory and stays byte-identical.
    std::optional<FusedRingConfig> fused_ring{std::nullopt};
    bool has_fused_ring() const { return fused_ring.has_value(); }
};

struct tensor_args_t {
    const Tensor& q;
    const Tensor& k;  // fused: the [B,1,T,D] persistent all-gather OUTPUT buffer the reader scores against
    const Tensor& weights;
    // Fused only: this device's per-chip LOCAL K shard [B,1,sll,D] = the all-gather INPUT. nullopt unfused.
    std::optional<Tensor> k_local{std::nullopt};
};

using tensor_return_value_t = Tensor;

using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::experimental::indexer_score
