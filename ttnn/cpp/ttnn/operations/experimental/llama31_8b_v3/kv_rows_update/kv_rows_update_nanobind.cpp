// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "kv_rows_update_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "kv_rows_update.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::experimental::llama31_8b_v3 {

void bind_kv_rows_update(nb::module_& llama31_8b_v3) {
    const auto* doc = R"doc(
        Writes the decode rows of one user into the paged K and V caches in one call, for Llama-3.1-8B decode on
        Blackhole P100.

        ``row_positions[j]`` is the ``update_idxs_tensor`` of slot ``j``: slot ``j`` of ``k`` / ``v`` is written to
        position ``row_positions[j][j]`` when it is ``>= 0``, otherwise skipped. The written positions must be
        consecutive in slot order. The page table row of user 0 is used. The caches are bit-identical to calling
        ``ttnn.experimental.paged_fused_update_cache`` once per slot with ``update_idxs_tensor=row_positions[j]``.
        Call ``kv_rows_update_supported`` first and fall back to that otherwise.

        Args:
            keys (ttnn.Tensor): [blocks, 8, 32, 128] bfloat8_b, TILE, DRAM interleaved, updated in place.
            k (ttnn.Tensor): [1, >=rows, 32, 128] bfloat16, TILE, L1 height sharded one slot per core.
            values (ttnn.Tensor): same as ``keys``.
            v (ttnn.Tensor): same as ``k``.
            row_positions (List[ttnn.Tensor]): 1, 2, 4 or 8 tensors, each [rows] int32, ROW_MAJOR, DRAM.
            page_table (ttnn.Tensor): [users, 1024] int32, ROW_MAJOR, DRAM.

        Returns:
            ttnn.Tensor: ``keys``.
        )doc";

    ttnn::bind_function<"kv_rows_update", "ttnn.experimental.llama31_8b_v3.">(
        llama31_8b_v3,
        doc,
        &kv_rows_update,
        nb::arg("keys"),
        nb::arg("k"),
        nb::arg("values"),
        nb::arg("v"),
        nb::arg("row_positions"),
        nb::arg("page_table"));

    const auto* doc_supported = R"doc(
        True when ``kv_rows_update`` accepts these tensors (see ``kv_rows_update``).
        )doc";

    ttnn::bind_function<"kv_rows_update_supported", "ttnn.experimental.llama31_8b_v3.">(
        llama31_8b_v3,
        doc_supported,
        &kv_rows_update_supported,
        nb::arg("keys"),
        nb::arg("k"),
        nb::arg("values"),
        nb::arg("v"),
        nb::arg("row_positions"),
        nb::arg("page_table"));
}

}  // namespace ttnn::experimental::llama31_8b_v3
