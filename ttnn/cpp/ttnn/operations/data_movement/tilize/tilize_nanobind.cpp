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
    // The BFLOAT16 NaN rule in this docstring tracks tenstorrent/tt-metal#31406; drop it once that issue is fixed.
    const auto* doc = R"doc(
        Changes the layout of the input tensor from ROW_MAJOR to TILE.

        A ROW_MAJOR input's last dimension must be a multiple of 32 and its second-to-last dimension a multiple of the
        tile height. ``ttnn.tilize_with_val_padding`` pads other shapes.

        A ``BFLOAT16``, ``FLOAT32``, ``INT32``, ``UINT32`` or ``UINT16`` TILE input whose tile height differs from that of
        ``tile`` is re-tiled into ``tile``. One of the two tile heights must divide the other, and an input whose
        second-to-last dimension is not a multiple of both tile heights must hold one matrix: every dimension before the
        last two is 1. An interleaved input needs an interleaved output. A sharded input must be in L1, with shard heights
        that are multiples of both tile heights, and ``sub_core_grids`` set to `None`.

        Args:
            input_tensor (ttnn.Tensor): the input tensor, on device.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration of the output. Defaults to `None`, which uses the input tensor's memory configuration. A sharded output, including the default one for a sharded input, needs a shard shape that is a multiple of the tile shape. When the input and the output are HEIGHT_SHARDED, WIDTH_SHARDED or BLOCK_SHARDED in L1 with the same memory layout, ``tile`` is 32x32 and ``sub_core_grids`` is `None`, the output keeps the input's shard spec; otherwise it follows ``memory_config``.
            dtype (ttnn.DataType, optional): Data type of the output. Defaults to `None`, which keeps the input dtype, or gives ``FLOAT32`` for a ``FP8_E4M3`` input. A ``BFLOAT16`` or ``FLOAT32`` input can give a ``BFLOAT16``, ``FLOAT32``, ``BFLOAT8_B`` or ``BFLOAT4_B`` output. For an integer input it must be `None` or the input's dtype.
            use_multicore (bool, optional): Whether to use multicore. Defaults to `True`. A sharded input needs `True`.
            use_low_perf (bool, optional): Whether to run on one core with one-tile buffers, which uses the least L1. It applies to ROW_MAJOR inputs other than ``UINT8`` on Blackhole. Defaults to `False`. A sharded input needs `False`.
            tile (ttnn.Tile, optional): Tile shape of the output. It must be 32 wide, with a height of 32, 16, 8, 4, 2 or 1. ``BFLOAT8_B`` and ``BFLOAT4_B`` outputs need a height of 32. Defaults to `ttnn.Tile((32, 32))`.
            sub_core_grids (ttnn.CoreRangeSet, optional): Cores to run the operation on. Defaults to `None`, which lets the operation choose its cores. With a grid of one core, the operation runs on that core and needs an interleaved or HEIGHT_SHARDED input, and on Blackhole an input other than ``UINT8``.

        Returns:
            ttnn.Tensor: the output tensor, in TILE layout.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, INT32, UINT32, UINT16, UINT8
                 - ROW_MAJOR

            ``BFLOAT16`` input values must not be NaN. On Blackhole, a ``UINT8`` input must be in DRAM and needs a tile height
            of 2 or more. On Quasar, a ``FLOAT32`` input needs a tile height of 32.

            On Blackhole, a ``FP8_E4M3`` ROW_MAJOR input is also supported. Its output defaults to ``FLOAT32`` and can be
            ``FLOAT32``, ``BFLOAT16``, ``BFLOAT8_B`` or ``BFLOAT4_B``, and its last dimension must be more than 32.

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded input: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED in L1; HEIGHT_SHARDED and WIDTH_SHARDED in DRAM
            - Sharded output: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED in L1; HEIGHT_SHARDED and WIDTH_SHARDED in DRAM
            - ND sharded: L1 input and output
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
