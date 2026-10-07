// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "select_tile_rows_nanobind.hpp"

#include "select_tile_rows.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

namespace ttnn::operations::experimental::kda::select_tile_rows::detail {

void bind_select_tile_rows(nb::module_& mod) {
    ttnn::bind_function<"select_tile_rows", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        Gather rows of a tiled tensor, chosen on device, into a row-major tensor.

        This is ``ttnn.embedding`` for a TILE-layout table: row ``indices[i]`` of the flattened
        ``[rows, columns]`` input becomes output row ``i``, bit for bit, without untilizing the table.

        Args:
            input (ttnn.Tensor): BFLOAT16 TILE-layout ``[1, rows, columns]`` interleaved table.
            indices (ttnn.Tensor): UINT32 row-major ``[n]`` row indices, at most 16, read on device.

        Keyword Args:
            width (int): Leading tile-aligned columns to gather.
            memory_config (ttnn.MemoryConfig, optional): Interleaved output memory. Defaults to DRAM.

        Returns:
            ttnn.Tensor: BFLOAT16 row-major ``[1, n, width]`` rows.
        )doc",
        &ttnn::experimental::kda::select_tile_rows,
        nb::arg("input").noconvert(),
        nb::arg("indices").noconvert(),
        nb::kw_only(),
        nb::arg("width"),
        nb::arg("memory_config") = nb::none());

    ttnn::bind_function<"select_history_rows", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        Gather a chronological history selection's rows, derived on device, into row-major tensors.

        The rows are the ones the ``chronological_selections`` record ``record`` lists, derived on device from
        ``actual_start`` (and ``actual_end``) without materializing the selection table.

        Args:
            input (ttnn.Tensor): BFLOAT16 ``[1, rows, columns]`` interleaved table, TILE or ROW_MAJOR.
            record (int): A history record of ``ttnn.experimental.kda._selection_layout``.
            actual_start (ttnn.Tensor): Replicated UINT32 row-major scalar, the chunk's first absolute position.
            sequence_parallel_axis (int): Mesh axis partitioning the sequence.
            local_rows (int): Physical rows per sequence-parallel rank.

        Keyword Args:
            width (int, optional): Leading tile-aligned columns to gather from a TILE input. Defaults to all; a
                ROW_MAJOR input selects whole rows.
            actual_end (ttnn.Tensor, optional): Replicated UINT32 scalar, the exclusive valid end.
            rows_per_output (int): Split the rows into two outputs of this many rows. 0 keeps one output.
            memory_config (ttnn.MemoryConfig, optional): Interleaved output memory. Defaults to DRAM.

        Returns:
            list[ttnn.Tensor]: BFLOAT16 row-major ``[1, n, width]`` rows, one or two tensors.
        )doc",
        &ttnn::experimental::kda::select_history_rows,
        nb::arg("input").noconvert(),
        nb::arg("record"),
        nb::arg("actual_start").noconvert(),
        nb::arg("sequence_parallel_axis"),
        nb::arg("local_rows"),
        nb::kw_only(),
        nb::arg("width") = nb::none(),
        nb::arg("actual_end") = nb::none(),
        nb::arg("rows_per_output") = 0,
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::kda::select_tile_rows::detail
