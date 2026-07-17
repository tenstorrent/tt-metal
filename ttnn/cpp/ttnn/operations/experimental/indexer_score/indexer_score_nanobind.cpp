// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "indexer_score_nanobind.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "indexer_score.hpp"

namespace ttnn::operations::experimental::indexer_score::detail {

void bind_indexer_score(nb::module_& mod) {
    nb::class_<IndexerScoreProgramConfig>(mod, "IndexerScoreProgramConfig")
        .def(
            nb::init<std::size_t, std::size_t, std::size_t>(),
            nb::kw_only(),
            nb::arg("q_chunk_size") = 32,
            nb::arg("k_chunk_size") = 32,
            nb::arg("head_group_size") = 1)
        .def_rw("q_chunk_size", &IndexerScoreProgramConfig::q_chunk_size)
        .def_rw("k_chunk_size", &IndexerScoreProgramConfig::k_chunk_size)
        .def_rw("head_group_size", &IndexerScoreProgramConfig::head_group_size);

    ttnn::bind_function<"indexer_score_dsa", "ttnn.experimental.">(
        mod,
        R"doc(
        DeepSeek-V3.2 DSA / GLM-5 lightning-indexer scorer.

        score[b, 0, s, t] = sum_h relu(q[b,h,s,:] . k[b,t,:]) * weights[b,h,s]

        ReLU(q.kT) gated by the learned per-head weights and summed over ALL Hi
        index heads into one shared selection row [B, 1, Sq, T]. For MiniMax M3's
        raw-dot, per-GQA-group indexer, use ``indexer_score_msa`` instead.

        Args:
            q: [B, Hi, Sq, D] bf16 or bfp8_b tiled (post non-interleaved RoPE)
            k: [B, 1, T, D] bf16 or bfp8_b tiled, single shared head
            weights: [B, Hi, Sq, 1] bf16 tiled learned per-head gates (scale
                pre-folded)
            chunk_start_idx: absolute global position of rank 0's query row 0
                (rank 0 = lowest cluster_axis coord; causality: key t visible to
                query s iff t <= chunk_start + s). OMIT on a mesh -> deduced as
                T - sp_ring*Sq (sp_ring = mesh extent along cluster_axis, whole
                mesh if unset). Single device: set to history + rank*Sq per rank.
            program_config: work-unit knobs (q_chunk_size, k_chunk_size,
                head_group_size; elements, tile-aligned). Defaults always fit
                L1; raise head_group_size (0 = all resident) for performance.
            compute_kernel_config: optional DeviceComputeKernelConfig. Only
                math_fidelity is honored (default: HiFi2, or LoFi when q and k
                are both bfloat8_b); fp32_dest_acc_en / dst_full_sync_en must
                stay false (the custom LLK is validated for bf16 DEST half-sync).
            cache_batch_idx: optional int. Selects the batch slot of a shared
                [B, 1, T, D] k cache (which may then be ND-sharded across DRAM
                banks). Re-applied each dispatch, so switching slots does not
                recompile.
            kv_len: optional int. Valid prefix of a k allocated at its full T;
                the rest is masked out. Tile-aligned, in (0, T], with
                chunk_start_idx + Sq <= kv_len. Re-applied each dispatch, so a
                serving loop growing kv_len (<= T) reuses one program -- no
                recompile. Only output columns [0, kv_len) are written.
            cluster_axis: mesh axis that is the SP ring. On a mesh, device r uses
                chunk_start = chunk_start_idx + r*Sq, where r is its linearized
                index along this axis (Sq = q seq len). None = linear device order
                (1 on a single device, so chunk_start_idx is used as-is).
            seq_subshard_axis: optional int, the SECOND mesh axis (TP) the query seq is
                block-cyclically sub-sharded over, on top of the SP block-cyclic layout.
                Set it alongside a named cluster_axis + block_cyclic when the query rows
                are further split across TP (block_cyclic_chunk_local == tp*q_isl): the
                exact geometry adds tp_rank*Sq to each device's block-cyclic position, so
                the causal mask stays correct under rotated (mid-slab) starts — unlike the
                cluster_axis=None flat path, which is linear-approximate there. Requires a
                block-cyclic layout and an axis distinct from cluster_axis.
            block_cyclic_sp_axis: optional int. The MESH AXIS the K cache was striped
                over (chunked prefill + SP all-gather leaves [B,1,T,D] k in per-SP-shard
                slab / block-cyclic physical order). ``sp`` is DERIVED from the mesh
                shape on that axis, so a caller cannot pass an sp that disagrees with
                the device. The reader reads the permuted cache back in natural token
                order (invP per tile) so scores come out token-ordered and the causal
                mask/pool stay exact. Unset (or sp==1) = contiguous K (no remap). Pair
                with block_cyclic_chunk_local. Interface matches ttnn.transformer.sparse_sdpa.
            block_cyclic_chunk_local: optional int, REQUIRED with block_cyclic_sp_axis.
                The per-shard chunk length (chunk_size_global / sp). Cross-checked
                against q: must equal q_isl (Sq, seq sharded only on the SP axis) or
                tp*q_isl (tp = mesh_size/sp). The tp*q_isl case with tp>1 (seq sharded
                across BOTH axes) has two forms: with cluster_axis=None it uses flat
                row-major linearization over all devices (linear-approximate under a
                mid-slab start); with a named cluster_axis it must also pass
                seq_subshard_axis naming the TP axis, which gives the rotation-exact 2D
                geometry (see seq_subshard_axis above).

        Returns: score [B, 1, Sq, T] bf16 row-major; future/pad columns -inf.
        )doc",
        &ttnn::experimental::indexer_score_dsa,
        nb::arg("q"),
        nb::arg("k"),
        nb::arg("weights"),
        nb::kw_only(),
        nb::arg("chunk_start_idx") = std::nullopt,
        nb::arg("program_config") = IndexerScoreProgramConfig{},
        nb::arg("compute_kernel_config") = std::nullopt,
        nb::arg("cache_batch_idx") = std::nullopt,
        nb::arg("kv_len") = std::nullopt,
        nb::arg("cluster_axis") = std::nullopt,
        nb::arg("seq_subshard_axis") = std::nullopt,
        nb::arg("block_cyclic_sp_axis") = std::nullopt,
        nb::arg("block_cyclic_chunk_local") = std::nullopt);

    ttnn::bind_function<"indexer_score_msa", "ttnn.experimental.">(
        mod,
        R"doc(
        MiniMax-M3 MSA lightning-indexer scorer.

        score[b, g, s, t] = sum_{h in group g} (q[b,h,s,:] . k[b,t,:]) * scale

        Raw dot product (no ReLU), no learned gates -- only a 1/sqrt(d) ``scale``
        (applied as a constant gate). The Hi index heads are partitioned into
        ``num_groups`` GQA groups and summed WITHIN each group only, giving one
        output plane per group (no cross-group sum) for per-group selection.
        For DeepSeek-V3.2 / GLM-5's relu + gated + head-summed indexer, use
        ``indexer_score_dsa`` instead.

        Args:
            q: [B, Hi, Sq, D] bf16 or bfp8_b tiled (post non-interleaved RoPE)
            k: [B, 1, T, D] bf16 or bfp8_b tiled, single shared head
            chunk_start_idx: absolute global position of rank 0's query row 0
                (causality: key t visible to query s iff t <= chunk_start + s).
                OMIT on a mesh -> deduced as T - sp_ring*Sq; single device: set to
                history + rank*Sq per rank (same semantics as indexer_score_dsa).
            scale: constant gate applied to every head (e.g. 1/sqrt(d))
            num_groups: REQUIRED (no default). 1 sums all Hi heads into one plane
                (the TP=4 group-aligned deployment, Hi=1/device). G>1 partitions the heads into G groups
                of Hi/G and sums within each group -> output [B, G, Sq, T] (multiple
                GQA groups on one chip). G>1 needs all heads resident
                (head_group_size 0 or Hi) and k_chunk_size >= 64.
            block_size: 0 (default) = no pooling -> score [B, G, Sq, T]. >0 =
                block-max-pool over block_size keys -> score [B, G, Sq, T/block_size]
                (block selection; the downstream topk then picks per-group top-k
                blocks). Requires block_size a multiple of 32, T % block_size == 0,
                and k_chunk_size % block_size == 0.
            program_config: work-unit knobs (q_chunk_size, k_chunk_size,
                head_group_size; elements, tile-aligned). Defaults always fit L1.
            compute_kernel_config: optional DeviceComputeKernelConfig. Only
                math_fidelity is honored (default: HiFi2, or LoFi when q and k
                are both bfloat8_b); fp32_dest_acc_en / dst_full_sync_en must
                stay false (the custom LLK is validated for bf16 DEST half-sync).
            cache_batch_idx: optional int. Selects the batch slot of a shared
                [B, 1, T, D] k cache (which may then be ND-sharded across DRAM
                banks). Re-applied each dispatch, so switching slots does not
                recompile. Same semantics as indexer_score_dsa.
            kv_len: optional int. Valid prefix of a k allocated at its full T;
                the rest is masked out. Tile-aligned, in (0, T], with
                chunk_start_idx + Sq <= kv_len. When block-max-pooling
                (block_size > 0) it must also be a multiple of block_size (whole
                blocks are written). Re-applied each dispatch (a serving loop
                growing kv_len reuses one program). Only columns/blocks within
                the valid prefix are written.
            cluster_axis: mesh axis that is the SP ring. On a mesh, device r uses
                chunk_start = chunk_start_idx + r*Sq, where r is its linearized
                index along this axis. None = linear device order.
            block_cyclic_sp_axis / block_cyclic_chunk_local: same semantics as
                indexer_score_dsa. block_cyclic_sp_axis = the MESH AXIS the K cache
                was striped over (sp derived from the mesh shape on that axis) and
                block_cyclic_chunk_local = the per-shard chunk length; the reader
                reads the permuted cache back in natural token order so the per-group
                scores (and the block-max-pool, which pools token-contiguous blocks)
                come out correct. Unset = contiguous K.

        Returns: score [B, num_groups, Sq, T_out] bf16 row-major (T_out = T, or
            T/block_size when block-max-pooling); future/pad columns/blocks -inf.
        )doc",
        &ttnn::experimental::indexer_score_msa,
        nb::arg("q"),
        nb::arg("k"),
        nb::kw_only(),
        nb::arg("num_groups"),  // required: per-GQA-group selection is MSA's purpose, no implicit default
        nb::arg("chunk_start_idx") = std::nullopt,
        nb::arg("scale") = 1.0f,
        nb::arg("block_size") = 0,
        nb::arg("program_config") = IndexerScoreProgramConfig{},
        nb::arg("compute_kernel_config") = std::nullopt,
        nb::arg("cache_batch_idx") = std::nullopt,
        nb::arg("kv_len") = std::nullopt,
        nb::arg("cluster_axis") = std::nullopt,
        nb::arg("block_cyclic_sp_axis") = std::nullopt,
        nb::arg("block_cyclic_chunk_local") = std::nullopt);

    ttnn::bind_function<"ring_indexer_score_dsa", "ttnn.experimental.">(
        mod,
        R"doc(
        Ring-fused DeepSeek-V3.2 DSA / GLM-5 lightning-indexer scorer.

        Identical score semantics to ``indexer_score_dsa`` but SUBSUMES the SP all-gather: rather than the
        caller pre-gathering K, it takes this chip's LOCAL K shard ``k_local`` [B,1,sll,D] (the all-gather
        input) plus a pre-allocated ``k`` [B,1,T,D] persistent buffer (the gather output). One program
        co-schedules the ring_attention all-gather with the indexer compute so fabric transport overlaps
        scoring; the reader gates each K band on ONLY the SP shards that band touches, so it scores already-
        arrived shards while farther slabs are still in flight. DSA only -- there is no fused MSA variant.

        Args:
            q: [B, Hi, Sq, D] bf16/bfp8_b tiled (post non-interleaved RoPE); see indexer_score_dsa
            k: [B, 1, T, D] bf16/bfp8_b tiled PERSISTENT all-gather OUTPUT buffer, T = sp*sll; the gather fills
                the remote SP shards in place (the local slab is read from k_local, not k)
            weights: [B, Hi, Sq, 1] bf16 tiled learned per-head gates (scale pre-folded)
            k_local: [B, 1, sll, D] bf16/bfp8_b tiled -- this chip's SP shard and the all-gather INPUT,
                sll = T/sp; must match k's dtype
            ag_multi_device_global_semaphore: list of the all-gather's out-ready global semaphores; requires
                >= 2 (the forward and backward ring directions)
            cluster_axis: mesh axis that is the SP ring -- both the gather axis and the causality axis.
                REQUIRED here (optional in indexer_score_dsa)
            topology: ttnn.Topology -- ttnn.Topology.Linear (non-torus grid) or ttnn.Topology.Ring
            num_links: int, fabric links for the gather (default 1)
            ag_sub_device_id: optional ttnn.SubDeviceId scoping the AG worker cores (kept disjoint from the
                compute grid so transport and compute cores do not collide)
            chunk_start_idx: optional int, rank 0's global query start; see indexer_score_dsa
            program_config: IndexerScoreProgramConfig work-unit knobs; see indexer_score_dsa.
                head_group_size must be 0 (all Hi resident) or Hi -- head streaming is not supported here
            compute_kernel_config: optional DeviceComputeKernelConfig (only math_fidelity honored)
            cache_batch_idx: optional int, batch slot of a shared K cache; see indexer_score_dsa
            kv_len: optional int, valid tile-aligned key prefix in (0, T]; see indexer_score_dsa
            seq_subshard_axis: optional int, 2D SP×TP -- the (TP) mesh axis the query rows are ALSO block-cyclic
                sub-sharded over, on top of the SP shard. The K cache stays SP-sharded + TP-replicated (the ring
                AG still gathers along cluster_axis), so only the causal query geometry gains the tp_rank*Sq
                sub-offset. nullopt = query sharded on the SP axis only. See indexer_score_dsa.
            block_cyclic_sp_axis: optional int, mesh axis the cache was striped over; MUST equal cluster_axis;
                see indexer_score_dsa
            block_cyclic_chunk_local: optional int, per-shard chunk length; required with block_cyclic_sp_axis

        Returns: score [B, 1, Sq, T] bf16 row-major; future/pad columns -inf.
        )doc",
        &ttnn::experimental::ring_indexer_score_dsa,
        nb::arg("q"),
        nb::arg("k"),
        nb::arg("weights"),
        nb::arg("k_local"),
        nb::arg("ag_multi_device_global_semaphore"),
        nb::kw_only(),
        nb::arg("cluster_axis"),
        nb::arg("topology"),
        nb::arg("num_links") = 1,
        nb::arg("ag_sub_device_id") = std::nullopt,
        nb::arg("chunk_start_idx") = std::nullopt,
        nb::arg("program_config") = IndexerScoreProgramConfig{},
        nb::arg("compute_kernel_config") = std::nullopt,
        nb::arg("cache_batch_idx") = std::nullopt,
        nb::arg("kv_len") = std::nullopt,
        nb::arg("seq_subshard_axis") = std::nullopt,
        nb::arg("block_cyclic_sp_axis") = std::nullopt,
        nb::arg("block_cyclic_chunk_local") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::indexer_score::detail
