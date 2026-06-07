// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_indexed_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "rotary_embedding_indexed.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::rotary_embedding_indexed::detail {

void bind_rotary_embedding_indexed(nb::module_& mod) {
    ttnn::bind_function<"rotary_embedding_indexed", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            KV-pad-aware indexed rotary embedding for chunked prefill.

            Applies rotary embedding to a per-chip input chunk, indexing into SP-sharded cos/sin
            caches at a per-device offset derived on-device from a single global valid-KV length
            (`kv_actual_global`) and the device's coordinate along `cluster_axis`. The boundary
            chip's older-then-wrap token layout is read with a single contiguous offset because the
            wrap is absorbed by the block-cyclic cos/sin shard layout.

            `kv_actual_global` is a per-call scalar held in a common runtime arg and patched on cache
            hits, so its value is out of the program hash and successive chunks reuse one cached program.

            Args:
                input (ttnn.Tensor): 4D per-chip input chunk on device, TILE layout
                    [1, n_heads, chunk_local, head_dim].
                cos (ttnn.Tensor): 4D cos cache on device, TILE layout, SP-sharded over
                    `cluster_axis` in block-cyclic order keyed by `chunk_local`.
                sin (ttnn.Tensor): 4D sin cache, same layout/shape as `cos`.
                trans_mat (ttnn.Tensor): rotation transformation matrix (one tile), replicated.
                kv_actual_global (int): prior valid global KV length in tokens (tile-aligned).
                cluster_axis (int): mesh axis the cos/sin caches are SP-sharded along (0 or 1).

            Returns:
                ttnn.Tensor: a new tensor with the same spec as `input`, rotary-embedded.
        )doc",
        &ttnn::operations::experimental::deepseek_prefill::rotary_embedding_indexed::rotary_embedding_indexed,
        nb::arg("input").noconvert(),
        nb::arg("cos").noconvert(),
        nb::arg("sin").noconvert(),
        nb::arg("trans_mat").noconvert(),
        nb::arg("kv_actual_global"),
        nb::arg("cluster_axis"),
        nb::arg("memory_config") = std::nullopt,
        nb::arg("compute_kernel_config") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::rotary_embedding_indexed::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_rotary_embedding_indexed(::nanobind::module_& mod) {
    rotary_embedding_indexed::detail::bind_rotary_embedding_indexed(mod);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
