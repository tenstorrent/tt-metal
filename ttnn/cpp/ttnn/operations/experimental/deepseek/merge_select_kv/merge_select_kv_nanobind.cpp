// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "merge_select_kv_nanobind.hpp"

#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deepseek/merge_select_kv/merge_select_kv.hpp"

namespace ttnn::operations::experimental::deepseek::detail {

void bind_merge_select_kv(::nanobind::module_& mod) {
    namespace nb = nanobind;
    ttnn::bind_function<"merge_select_kv", "ttnn.experimental.deepseek.">(
        mod,
        R"doc(
        Second half of a key-split ``fused_lightning_select_kv``: merges the per-device candidates
        into the global top-k and gathers those rows from this device's ``kv_cache``.

        ``candidates`` is the ``num_parts > 1`` output of ``fused_lightning_select_kv`` from every
        device, concatenated along dim 2 in part order, e.g.
        ``ttnn.all_gather(candidates, dim=2, cluster_axis=...)``. The merge keeps the k largest
        scores over every part, ties to the lower key, which is the unsplit op's selection, and
        writes the rows in ascending key order like the unsplit op.

        Args:
            candidates (ttnn.Tensor): ``[1, 1, num_parts, 2k + 16]`` INT32 row-major interleaved.
            kv_cache (ttnn.Tensor): row-major interleaved ``[num_blocks, 1, block_size, Dh]``; the
                candidates' rows index it.
            k (int): number of selected rows.

        Keyword Args:
            memory_config (Optional[ttnn.MemoryConfig]): output memory config, interleaved.
                Defaults to interleaved L1.
            output_tensor (Optional[ttnn.Tensor]): preallocated ``[1, 1, rows, Dh]`` row-major
                interleaved output with ``kv_cache``'s dtype. Only rows
                ``[output_row_offset, output_row_offset + k)`` are written.
            output_row_offset (int): output row of selected row 0. Requires ``output_tensor``.

        Returns:
            ttnn.Tensor: the selected rows ``[1, 1, k, Dh]``, or ``output_tensor``. With fewer than
            k candidates in total, only that many rows are written.
        )doc",
        &ttnn::experimental::deepseek::merge_select_kv,
        nb::arg("candidates"),
        nb::arg("kv_cache"),
        nb::arg("k"),
        nb::kw_only(),
        nb::arg("memory_config") = std::nullopt,
        nb::arg("output_tensor") = std::nullopt,
        nb::arg("output_row_offset") = 0);
}

}  // namespace ttnn::operations::experimental::deepseek::detail
