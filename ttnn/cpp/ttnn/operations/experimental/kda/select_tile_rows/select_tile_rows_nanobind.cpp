// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "select_tile_rows_nanobind.hpp"

#include "select_tile_rows.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>

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
}

}  // namespace ttnn::operations::experimental::kda::select_tile_rows::detail
