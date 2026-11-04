// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "tilize.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_tilize(nb::module_& mod) {
    const auto* doc = R"doc(
        Changes the layout of the input tensor from ROW_MAJOR to TILE.

        The input is not padded: its last dimension must be a multiple of 32 and its second-to-last dimension a
        multiple of the tile height. Use ``ttnn.tilize_with_val_padding`` for other shapes.

        Args:
            input_tensor (ttnn.Tensor): the input tensor, on device.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`, which uses the input tensor's memory configuration.
            dtype (ttnn.DataType, optional): Data type of the output tensor. Defaults to `None`, which keeps the input dtype.
            use_multicore (bool, optional): Whether to use multicore. Defaults to `True`. Sharded inputs need `True`.
            use_low_perf (bool, optional): Use a low performance version that uses less memory. USE ONLY IF ABSOLUTELY NEEDED IN MODELS. Defaults to `False`.
            tile (ttnn.Tile, optional): Tile shape for the output tensor. It must be 32 wide, with a height of 32, 16, 8, 4, 2 or 1. Defaults to `ttnn.Tile((32, 32))`.
            sub_core_grids (ttnn.CoreRangeSet, optional): Cores to run the operation on. Defaults to `None`, which uses the whole device.

        Returns:
            ttnn.Tensor: the output tensor, in TILE layout.

        Note:
            Supported dtypes and layouts for the input tensor:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, INT32, UINT32, UINT16, UINT8
                 - ROW_MAJOR

            ``dtype`` can convert the output, for example from ``BFLOAT16`` input to ``BFLOAT8_B`` output.
            ``BFLOAT8_B`` and ``BFLOAT4_B`` outputs need a tile height of 32.

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED input in L1 or DRAM; the output can use the same sharding or be interleaved
            - ND sharded: input and output
    )doc";

    ttnn::bind_function<"tilize">(
        mod,
        doc,
        &ttnn::tilize,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("dtype") = nb::none(),
        nb::arg("use_multicore") = true,
        nb::arg("use_low_perf") = false,
        nb::arg("tile") = tt::tt_metal::Tile(),  // default to 32x32 tile
        nb::arg("sub_core_grids") = nb::none());
}
}  // namespace ttnn::operations::data_movement::detail
