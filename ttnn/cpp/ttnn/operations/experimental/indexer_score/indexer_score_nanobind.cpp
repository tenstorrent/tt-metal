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

    ttnn::bind_function<"indexer_score", "ttnn.experimental.">(
        mod,
        R"doc(
        DeepSeek-V3.2 DSA lightning-indexer scorer.

        score[b, s, t] = sum_h relu(q[b,h,s,:] . k[b,t,:]) * weights[b,h,s]

        Args:
            q: [B, Hi, Sq, D] bf16 or bfp8_b tiled (post non-interleaved RoPE)
            k: [B, 1, T, D] bf16 or bfp8_b tiled, single shared head
            weights: [B, Hi, Sq, 1] bf16 tiled, scales pre-folded
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

        Returns: score [B, 1, Sq, T] bf16 row-major; future/pad columns -inf.
        )doc",
        &ttnn::experimental::indexer_score,
        nb::arg("q"),
        nb::arg("k"),
        nb::arg("weights"),
        nb::kw_only(),
        nb::arg("chunk_start_idx") = std::nullopt,
        nb::arg("program_config") = IndexerScoreProgramConfig{},
        nb::arg("compute_kernel_config") = std::nullopt,
        nb::arg("cache_batch_idx") = std::nullopt,
        nb::arg("kv_len") = std::nullopt,
        nb::arg("cluster_axis") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::indexer_score::detail
