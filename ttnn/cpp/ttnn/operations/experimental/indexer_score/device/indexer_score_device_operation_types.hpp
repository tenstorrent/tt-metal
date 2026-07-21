// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa/device/block_cyclic_layout.hpp"  // ttnn::prim::BlockCyclicLayout (shared)
#include <tt-metalium/base_types.hpp>

namespace ttnn::operations::experimental::indexer_score {

// Work-unit knobs (elements, tile-aligned; SDPAProgramConfig analogue).
// One unit = q_chunk rows x k_chunk keys; heads stream in head_group blocks.
struct IndexerScoreProgramConfig {
    std::size_t q_chunk_size = 32;
    std::size_t k_chunk_size = 32;
    std::size_t head_group_size = 1;  // heads resident at once; 1 always fits L1, raise for perf (0 = all)
};

// Resolve head_group_size to a concrete head count (0 = all Hi). Single-sourced so validate and the
// factory can't drift on the "0 = all" contract.
inline uint32_t resolve_head_group(const IndexerScoreProgramConfig& cfg, uint32_t Hi) {
    return cfg.head_group_size == 0 ? Hi : static_cast<uint32_t>(cfg.head_group_size);
}

// Shared type (sp, chunk_local) with sparse_sdpa / sparse_sdpa_msa — see block_cyclic_layout.hpp; the
// indexer reader reads K in logical order via this invP (reader_indexer_score.cpp).
using ttnn::prim::BlockCyclicLayout;

struct operation_attributes_t {
    // Absolute chunk_start of rank 0. Rank r uses chunk_start_idx + r*Sq; the per-device value is derived
    // host-side and passed to compute as a RUNTIME arg (hash-excluded), so distinct values reuse one program.
    uint32_t chunk_start_idx{0};  // elements, tile-aligned
    // Mesh axes the query sequence is sharded over, outermost (SP ring) first: {} = linear device order,
    // {sp} = 1D SP ring, {sp, tp} = 2D SP ring + TP sub-shard. The SP axis sets each device's causal offset;
    // the optional TP axis (only alongside an SP axis + block_cyclic) adds a Sq-row sub-offset so each device
    // owns [tp_rank*Sq, (tp_rank+1)*Sq) of its SP chip's chunk_local slab. Read via sp_axis()/tp_axis().
    // Hashed via those accessors (it shapes the causal geometry, so distinct shardings get distinct programs).
    std::vector<uint32_t> seq_shard_axes{};
    std::optional<uint32_t> sp_axis() const {
        return seq_shard_axes.empty() ? std::nullopt : std::optional<uint32_t>(seq_shard_axes.front());
    }
    std::optional<uint32_t> tp_axis() const {
        return seq_shard_axes.size() >= 2 ? std::optional<uint32_t>(seq_shard_axes[1]) : std::nullopt;
    }
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
};

struct tensor_args_t {
    const Tensor& q;
    const Tensor& k;
    const Tensor& weights;
};

using tensor_return_value_t = Tensor;

using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::experimental::indexer_score
