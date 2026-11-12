// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "indexed_entry_copy_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/kda/indexed_entry_copy/indexed_entry_copy.hpp"

namespace ttnn::operations::experimental::kda::indexed_entry_copy::detail {

void bind_indexed_entry_copy(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Copies one dim-0 entry between a multi-entry tensor and a single-entry tensor, in place into
            ``destination``. The entry is ``index[0] * index_scale + index_offset``, read on device when the
            operation runs, so a captured trace follows whatever is written into ``index`` before each replay.
            An entry at or past the multi-entry side's dimension 0 writes nothing. When both tensors hold one
            entry, entry 0 copies the whole tensor.

            Either tensor may be interleaved DRAM; a TILE tensor may also be ND-sharded DRAM.

            It moves one KDA carry between a model's scratch and the slot-indexed migration slabs while the KDA
            operations are unaware of slots; it is expected to go away if they become slot-aware. With
            ``source_part`` each device stores only its own heads of a carry replicated across sequence-parallel
            ranks, without a separate partition.

            Args:
                source (ttnn.Tensor): Tensor read from.
                destination (ttnn.Tensor): Tensor written in place; same dtype, layout, tile, page size and
                    shape past dim 0 as ``source``. At most one of the two may hold more than one entry.
                index (ttnn.Tensor): One-element UINT32 ROW_MAJOR interleaved device tensor.

            Keyword Args:
                index_scale (int): Multiplier applied to ``index[0]``; at least 1.
                index_offset (int): Added after scaling.
                zero_fill_at_zero (ttnn.Tensor, optional): One-element UINT32 ROW_MAJOR interleaved device tensor,
                    read on device like ``index``. When it reads 0 the destination entry is filled with zeros, in
                    either direction, without reading ``source``; any other value copies as usual. Defaults to None,
                    which always copies.
                source_part (ttnn.Tensor, optional): One-element UINT32 ROW_MAJOR interleaved device tensor, read on
                    device like ``index``; its value may differ per device of a mesh. When given, the single-entry
                    ``source`` holds ``P`` entries' worth of ``destination`` along dim 1 (its dim 1 is ``P`` times
                    the destination's; every other dim past 0 matches), and the copy reads part ``source_part[0]`` of
                    it: dim 1 of the source is ``source_part_groups`` equal groups, each holding the ``P`` parts in
                    order, and the copied entry takes that part from every group. A part at or past ``P`` writes
                    nothing. Dim 1 must span whole pages: rank at least 4 for TILE, 3 for ROW_MAJOR. Defaults to None,
                    which copies the whole source.
                source_part_groups (int): Groups of parts along dim 1 of a partitioned source; at least 1, and 1
                    without ``source_part``. Defaults to 1.

            Returns:
                ttnn.Tensor: ``destination``.
        )doc";

    ttnn::bind_function<"indexed_entry_copy", "ttnn.experimental.kda.">(
        mod,
        doc,
        &ttnn::experimental::kda::indexed_entry_copy,
        nb::arg("source").noconvert(),
        nb::arg("destination").noconvert(),
        nb::arg("index").noconvert(),
        nb::kw_only(),
        nb::arg("index_scale"),
        nb::arg("index_offset"),
        nb::arg("zero_fill_at_zero") = nb::none(),
        nb::arg("source_part") = nb::none(),
        nb::arg("source_part_groups") = 1);
}

}  // namespace ttnn::operations::experimental::kda::indexed_entry_copy::detail
