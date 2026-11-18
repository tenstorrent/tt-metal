// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/fold/fold_nanobind.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <variant>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn/operations/data_movement/fold/fold.hpp"
#include "ttnn/operations/data_movement/fold/device/fold_device_op.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

void bind_fold_operation(nb::module_& mod) {
    const auto* doc = R"doc(
        Moves each ``stride_h x stride_w`` block of pixels of an ``[N, H, W, C]`` tensor into the channel dimension
        (space-to-depth). `padding` first pads the input with zeros to ``[N, Hp, Wp, Cp]``. Output pixel ``(n, h, w)``
        then holds pixel ``(n, h * stride_h + i, w * stride_w + j)`` of the padded input in channels
        ``(i * stride_w + j) * Cp`` to ``(i * stride_w + j + 1) * Cp - 1``, for every ``i`` below `stride_h` and every
        ``j`` below `stride_w`.

        Args:
            input (ttnn.Tensor): the input tensor, of shape ``[N, H, W, C]``.
            stride_h (int): the height of each block. Must be greater than 0 and divide ``Hp``.
            stride_w (int): the width of each block. Must be greater than 0 and divide ``Wp``.
            use_transpose_as_fold (bool, optional): Must be `False`. Defaults to `False`.
            output_shape (ttnn.Shape, optional): Must be `None`. Defaults to `None`.
            padding (List[int], optional): Zero padding applied before the fold: ``[pad_h, pad_w]`` pads both sides of ``H`` by ``pad_h`` and both sides of ``W`` by ``pad_w``; ``[pad_top, pad_bottom, pad_left, pad_right]`` pads each side of ``H`` and ``W``; ``[pad_top, pad_bottom, pad_left, pad_right, pad_c_front, pad_c_back]`` also pads both ends of ``C``. For a TILE input every value must be 0. Defaults to `[0, 0]`.
            grid_size (ttnn.CoreRangeSet, optional): Must be `None`. Defaults to `None`.
            override_memory_config (ttnn.MemoryConfig, optional): Memory configuration of the output: interleaved in DRAM or L1, or HEIGHT_SHARDED, WIDTH_SHARDED or BLOCK_SHARDED in L1. For a sharded configuration without a shard spec, the operation chooses one. Defaults to `None`, which keeps an interleaved input's memory configuration and gives a HEIGHT_SHARDED input a HEIGHT_SHARDED L1 output.
            collapse_output (bool, optional): Whether to merge the batch and spatial dimensions of the output, giving it shape ``[1, 1, N * (Hp / stride_h) * (Wp / stride_w), Cp * stride_h * stride_w]``. Defaults to `False`.

        Returns:
            ttnn.Tensor: the output tensor, of shape ``[N, Hp / stride_h, Wp / stride_w, Cp * stride_h * stride_w]``, in ROW_MAJOR layout with the input's dtype. A ``BFLOAT8_B`` input gives a ``BFLOAT16`` output.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, UINT16
                 - TILE, ROW_MAJOR
               * - BFLOAT8_B
                 - TILE

            ``BFLOAT16`` TILE input values must not be NaN.

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded: HEIGHT_SHARDED in L1, with a shard width equal to ``C``. A ROW_MAJOR input needs ``C`` and ``Cp`` times the element size to be multiples of 16 bytes. With ``H`` or ``W`` padding, a device opened with a non-zero ``l1_small_size`` is needed.
    )doc";

    ttnn::bind_function<"fold">(
        mod,
        doc,
        &ttnn::fold,
        nb::arg("input"),
        nb::arg("stride_h"),
        nb::arg("stride_w"),
        nb::arg("use_transpose_as_fold") = false,
        nb::arg("output_shape") = nb::none(),
        nb::arg("padding") = std::array<uint32_t, 2>{0, 0},
        nb::arg("grid_size") = nb::none(),
        nb::arg("override_memory_config") = nb::none(),
        nb::arg("collapse_output") = false);

    // Test-only hooks: _prim_fold bypasses the composite gate so validate_fold's FATAL surfaces;
    // _is_tile_native_fold_supported lets tests derive fits/over shapes from the same predicate.
    mod.def(
        "_prim_fold",
        [](const ttnn::Tensor& input, uint32_t stride_h, uint32_t stride_w, bool collapse_output) {
            return ttnn::prim::fold(input, stride_h, stride_w, collapse_output);
        },
        nb::arg("input"),
        nb::arg("stride_h"),
        nb::arg("stride_w"),
        nb::arg("collapse_output") = false);
    mod.def(
        "_is_tile_native_fold_supported",
        &ttnn::operations::data_movement::is_tile_native_fold_supported,
        nb::arg("input"),
        nb::arg("stride_h"),
        nb::arg("stride_w"));
}

}  // namespace ttnn::operations::data_movement
