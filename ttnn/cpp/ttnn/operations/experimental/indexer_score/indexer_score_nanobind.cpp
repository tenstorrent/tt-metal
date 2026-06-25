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
            chunk_start_idx: global position of query row 0 (causality: key t
                visible to query s iff t <= chunk_start_idx + s)
            program_config: work-unit knobs (q_chunk_size, k_chunk_size,
                head_group_size; elements, tile-aligned). Defaults always fit
                L1; raise head_group_size (0 = all resident) for performance.
            compute_kernel_config: optional DeviceComputeKernelConfig. Only
                math_fidelity is honored (default: HiFi2, or LoFi when q and k
                are both bfloat8_b); fp32_dest_acc_en / dst_full_sync_en must
                stay false (the custom LLK is validated for bf16 DEST half-sync).

        Returns: score [B, 1, Sq, T] bf16 row-major; future/pad columns -inf.
        )doc",
        &ttnn::experimental::indexer_score_dsa,
        nb::arg("q"),
        nb::arg("k"),
        nb::arg("weights"),
        nb::kw_only(),
        nb::arg("chunk_start_idx") = 0,
        nb::arg("program_config") = IndexerScoreProgramConfig{},
        nb::arg("compute_kernel_config") = std::nullopt);

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
            chunk_start_idx: global position of query row 0 (causality: key t
                visible to query s iff t <= chunk_start_idx + s)
            scale: constant gate applied to every head (e.g. 1/sqrt(d))
            num_groups: 1 sums all Hi heads into one plane (the TP=4 group-aligned
                deployment, Hi=1/device). G>1 partitions the heads into G groups
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
            chunk_offset: optional per-device causal chunk-start tensor (uint32,
                one 32x32 TILE per device; element [0,0] = chunk-start in TILES).
                When bound, the reader DRAM-reads each device's tile and the
                compute/writer kernels use that RUNTIME value as the causal-mask
                diagonal / local-block base instead of chunk_start_idx (lets each
                SP chip mask against its own absolute query positions). When None
                (default), the compile-time chunk_start_idx is used (single-shot).

        Returns: score [B, num_groups, Sq, T_out] bf16 row-major (T_out = T, or
            T/block_size when block-max-pooling); future/pad columns/blocks -inf.
        )doc",
        &ttnn::experimental::indexer_score_msa,
        nb::arg("q"),
        nb::arg("k"),
        nb::kw_only(),
        nb::arg("chunk_start_idx") = 0,
        nb::arg("scale") = 1.0f,
        nb::arg("num_groups") = 1,
        nb::arg("block_size") = 0,
        nb::arg("program_config") = IndexerScoreProgramConfig{},
        nb::arg("compute_kernel_config") = std::nullopt,
        nb::arg("chunk_offset") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::indexer_score::detail
