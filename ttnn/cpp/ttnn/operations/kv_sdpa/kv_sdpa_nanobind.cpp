// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "kv_sdpa_nanobind.hpp"

#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/kv_sdpa/kv_sdpa.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::kv_sdpa {

void bind_kv_sdpa_operation(nb::module_& mod) {
    ttnn::bind_function<"kv_sdpa">(
        mod,
        R"doc(kv_sdpa(q, k, v, *, attn_mask=None, scale=None, compute_kernel_config=None) -> ttnn.Tensor

        Specialized fused-flash scaled-dot-product attention for the small-query MQA case: Q length is
        one tile (32), K/V have a single (or grouped) KV head shared across Q heads, and attention is
        non-causal full attention. One core per Q head runs the transformer-SDPA online-softmax flash
        loop, specialized to this shape.

        Q: [1, NQH, 32, DH]; K/V: [1, NKH, KV, DH] with NKH dividing NQH. Output: [1, NQH, 32, DH].
        attn_mask (optional, [1,1,Sq,KV] bf16 additive mask over the full folded KV) is applied when provided; omit it for the fast unmasked path.

        max_kv_chunk_tiles (default 128) bounds tiles per flash K/V chunk (Sk_chunk_t * DHt). Larger
        means fewer chunks and less per-chunk overhead, but the double-buffered per-chunk K/V CBs grow
        linearly -- at DH=256 (DHt=8) the default yields 256-tile prefix CBs (~272 KB each). Lower it
        when the caller pins significant L1, or the CBs clash with those buffers.

        kv_splits (default 1) splits the PREFIX K/V across this many cores per Q head (flash-decode
        style), using NQH * kv_splits cores instead of NQH. Since Sq is a single tile there is no
        Q-parallelism, so this is the only way to use more than NQH cores. Split 0 also takes the
        suffix and reduces the partial (max, sum, out) states. Requires prefix_Kt % kv_splits == 0.
        )doc",
        &ttnn::kv_sdpa,
        nb::arg("q"),
        nb::arg("k"),
        nb::arg("v"),
        nb::kw_only(),
        nb::arg("attn_mask") = nb::none(),
        nb::arg("scale") = nb::none(),
        nb::arg("past_k") = nb::none(),
        nb::arg("past_v") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("max_kv_chunk_tiles") = 128,
        nb::arg("kv_splits") = 1,
        nb::arg("prefix_valid_tiles") = std::vector<uint32_t>{});
}

}  // namespace ttnn::operations::kv_sdpa
