// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "indexed_entry_copy_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/indexed_entry_copy/indexed_entry_copy.hpp"

namespace ttnn::operations::experimental::indexed_entry_copy::detail {

void bind_experimental_indexed_entry_copy_operation(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Copies one entry between a multi-entry tensor and a single-entry tensor, in place into
            ``destination``, with the entry chosen on device.

            Dimension 0 counts entries. Exactly one of ``source`` and ``destination`` may hold more than
            one entry; the other holds one. The copied entry is
            ``index[0] * index_scale + index_offset``, read on device from ``index`` when the operation
            runs, so a captured trace follows whatever is written into ``index`` before each replay.
            With an entry at or past the multi-entry tensor's dimension 0 the operation writes nothing.
            When both tensors hold one entry, entry 0 copies the whole tensor.

            An entry is the contiguous range of pages (tiles for TILE layout, rows for ROW_MAJOR) of its
            dimension-0 slice. Pages are addressed logically on both sides, so either tensor may be
            interleaved or ND-sharded DRAM; the physical layouts need not match.

            Args:
                source (ttnn.Tensor): Tensor read from.
                destination (ttnn.Tensor): Tensor written in place. Same dtype, layout, page size and
                    dimensions past 0 as ``source``.
                index (ttnn.Tensor): One-element UINT32 or INT32 ROW_MAJOR interleaved DRAM tensor.

            Keyword Args:
                index_scale (int): Multiplier applied to ``index[0]``; at least 1. Defaults to 1.
                index_offset (int): Added after scaling. Defaults to 0.

            Returns:
                ttnn.Tensor: ``destination``.
        )doc";

    ttnn::bind_function<"indexed_entry_copy", "ttnn.experimental.">(
        mod,
        doc,
        &ttnn::experimental::indexed_entry_copy,
        nb::arg("source").noconvert(),
        nb::arg("destination").noconvert(),
        nb::arg("index").noconvert(),
        nb::kw_only(),
        nb::arg("index_scale") = 1,
        nb::arg("index_offset") = 0);
}

}  // namespace ttnn::operations::experimental::indexed_entry_copy::detail
