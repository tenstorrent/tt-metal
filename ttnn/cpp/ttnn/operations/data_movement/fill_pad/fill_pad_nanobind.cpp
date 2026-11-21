// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fill_pad_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "fill_pad.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::data_movement {
namespace {

void bind_fill_pad_op(nb::module_& mod) {
    const auto* doc = R"doc(
        Writes `fill_value` into the tile padding of a TILE tensor: the elements past the tensor's shape in its last
        two dimensions, up to the next multiple of 32. The values inside the tensor's shape are unchanged. A tensor
        whose last two dimensions are multiples of 32 has no tile padding and comes back unchanged.

        Args:
            input_tensor (ttnn.Tensor): the input tensor, on device, of rank 2 or more, in TILE layout with 32x32 tiles.
            fill_value (float or int): the value to write into the tile padding. It must not be NaN. A ``BFLOAT16``, ``FLOAT32``, ``BFLOAT8_B`` or ``BFLOAT4_B`` tensor holds it at the precision of its dtype. For ``INT32``, ``UINT32`` and ``UINT16`` tensors it must be an int that the dtype can hold, and a negative ``INT32`` value must be at least ``-2**24``.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration of the output. Must be `None` or the memory configuration of `input_tensor`. Defaults to `None`, which uses the memory configuration of `input_tensor`.
            sub_core_grids (ttnn.CoreRangeSet, optional): Cores to run the operation on. It needs an interleaved input. Defaults to `None`, which lets the operation choose its cores.

        Returns:
            ttnn.Tensor: the output tensor, with the shape, dtype, layout, memory configuration and values of `input_tensor`, and `fill_value` in its tile padding.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, BFLOAT8_B, BFLOAT4_B, INT32, UINT32, UINT16
                 - TILE

            ``BFLOAT16`` input values must not be NaN.

            ``BFLOAT8_B`` and ``BFLOAT4_B`` tensors need a last dimension that is a multiple of 16, or a `fill_value`
            of 0.

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded: HEIGHT_SHARDED and WIDTH_SHARDED in L1, and BLOCK_SHARDED in L1 with a ROW_MAJOR shard orientation, for a tensor whose dimensions other than the last two are all 1
    )doc";

    ttnn::bind_function<"fill_implicit_tile_padding">(
        mod,
        doc,
        &ttnn::fill_implicit_tile_padding,
        nb::arg("input_tensor"),
        nb::arg("fill_value"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("sub_core_grids") = nb::none());
}

}  // namespace

void bind_fill_pad(nb::module_& mod) { bind_fill_pad_op(mod); }

}  // namespace ttnn::operations::data_movement
