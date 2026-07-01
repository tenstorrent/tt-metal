// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "indexer_score_nanobind.hpp"

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
            kv_len: optional int. The valid key prefix (= the real ISL); keys at
                positions >= kv_len are masked out. In (0, T]; need NOT be
                tile-aligned -- a sub-tile kv_len masks the pad columns of its last
                (partial) valid tile per-column. The causal window MAY extend past
                kv_len -- with a padded chunk (real
                ISL < the fixed chunk) the padded-query devices reach past the
                valid keys, which is fine (those keys are masked, their output
                discarded). Re-applied each dispatch, so a serving loop growing
                kv_len (<= T) reuses one program -- no recompile. Only output
                columns [0, kv_len) are meaningful.
            cluster_axis: mesh axis that is the SP ring. On a mesh, device r uses
                chunk_start = chunk_start_idx + r*Sq, where r is its linearized
                index along this axis (Sq = q seq len). None = linear device order
                (1 on a single device, so chunk_start_idx is used as-is).
            slab_sp: optional int. The SP the K cache slab was gathered across --
                i.e. the cache's OWN sequence-parallel degree, independent of how
                this op splits Q. The gathered [B,1,T,D] k is then in per-SP-shard
                slab (block-cyclic) physical order, and the reader reads it back in
                natural token order (invP per tile) so scores come out token-ordered
                and the causal mask/pool stay exact. Unset (or 1) = contiguous K (no
                remap). Pair with slab_chunk_size. NOT derived from the mesh -- the
                cache's slab is decoupled from this op's parallelization (e.g. an
                SP=8 cache, slab_sp=8, scored by an SP=32 indexer with a smaller Sq).
            slab_chunk_size: optional int, REQUIRED with slab_sp. The global prefill
                chunk granularity in tokens (e.g. 5120); the cache's per-shard slab
                width is slab_chunk_size / slab_sp. Must be divisible by slab_sp with
                a tile-aligned per-shard width, divide T, and be >= Sq.
            mid_slab_boundary: SINGLE-CHIP TESTING ONLY. Simulate this one device as
                the boundary chip whose Sq queries straddle a cache-slab boundary
                (the causal diagonal jumps mid-tensor). On a mesh the straddle is
                derived from the per-device position, so this is ignored there.

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
        nb::arg("slab_sp") = std::nullopt,
        nb::arg("slab_chunk_size") = std::nullopt,
        nb::arg("mid_slab_boundary") = false);

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
            cluster_axis: mesh axis that is the SP ring. On a mesh, device r uses
                chunk_start = chunk_start_idx + r*Sq, where r is its linearized
                index along this axis. None = linear device order.
            slab_sp / slab_chunk_size / mid_slab_boundary: same semantics as
                indexer_score_dsa. slab_sp = the SP the K cache slab was gathered
                across (the cache's own SP, decoupled from this op's split) and
                slab_chunk_size = the global chunk granularity; the reader reads the
                permuted cache back in natural token order so the per-group scores
                (and the block-max-pool, which pools token-contiguous blocks) come
                out correct. Unset = contiguous K. mid_slab_boundary is single-chip
                testing only.

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
        nb::arg("cluster_axis") = std::nullopt,
        nb::arg("slab_sp") = std::nullopt,
        nb::arg("slab_chunk_size") = std::nullopt,
        nb::arg("mid_slab_boundary") = false);
}

}  // namespace ttnn::operations::experimental::indexer_score::detail
