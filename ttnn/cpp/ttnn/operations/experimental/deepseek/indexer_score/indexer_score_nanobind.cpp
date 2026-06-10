// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "indexer_score_nanobind.hpp"

#include "ttnn-nanobind/bind_function.hpp"
#include "indexer_score.hpp"

namespace ttnn::operations::experimental::deepseek::indexer::detail {

void bind_indexer_score(nb::module_& mod) {
    ttnn::bind_function<"indexer_score", "ttnn.experimental.deepseek.">(
        mod,
        R"doc(
        DeepSeek-V3.2 DSA lightning-indexer scorer.

        score[b, s, t] = sum_h relu(q[b,h,s,:] . k[b,t,:]) * weights[b,h,s]

        Args:
            q: [B, Hi, Sq, D] bf16 tiled (post non-interleaved RoPE)
            k: [B, 1, T, D] bf16 tiled, single shared head
            weights: [B, Hi, Sq, 1] bf16 tiled, scales pre-folded
            is_causal: apply causality from chunk_start_idx
            chunk_start_idx: global position of query row 0

        Returns: score [B, 1, Sq, T] bf16 row-major; future/pad columns -inf.
        )doc",
        &ttnn::experimental::deepseek::indexer_score,
        nb::arg("q"),
        nb::arg("k"),
        nb::arg("weights"),
        nb::kw_only(),
        nb::arg("is_causal") = true,
        nb::arg("chunk_start_idx") = 0);
}

}  // namespace ttnn::operations::experimental::deepseek::indexer::detail
