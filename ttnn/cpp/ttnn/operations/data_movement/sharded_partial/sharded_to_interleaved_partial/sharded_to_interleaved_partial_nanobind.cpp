// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sharded_to_interleaved_partial_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "sharded_to_interleaved_partial.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

void bind_sharded_to_interleaved_partial(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Writes a sharded slice into rows of the interleaved TILE tensor `cache_tensor`, and returns `cache_tensor`.

        `cache_tensor` of shape ``[N, C, H, W]`` is read as ``N*C*H`` rows of width ``W``, split into `num_slices`
        equal slices of ``R = N*C*H / num_slices`` rows. The operation writes `input_tensor` into rows
        ``slice_index * R`` to ``(slice_index + 1) * R - 1`` and leaves the other rows of `cache_tensor` unchanged.
        With the same `num_slices` and `slice_index`, it writes the output of ``ttnn.interleaved_to_sharded_partial``
        back into the rows that output was copied from.

        Args:
            input_tensor (ttnn.Tensor): the slice to write, of shape ``[1, 1, R, W]``, in TILE layout with 32x32 tiles, sharded in L1, see Memory Support.
            cache_tensor (ttnn.Tensor): the interleaved tensor to write into, of shape ``[N, C, H, W]``, in TILE layout with 32x32 tiles, on the device of `input_tensor`. ``H`` and ``W`` must be multiples of 32. Its dtype must be that of `input_tensor` or one the slice is converted to: an input of BFLOAT16, FLOAT32 or BFLOAT8_B can be converted to any of these three dtypes. Input values must not be NaN when the dtypes differ.
            num_slices (int): number of slices to split the rows of `cache_tensor` into. Must be at least 1, and ``N*C*H / num_slices`` must be a multiple of 32.
            slice_index (int): index of the slice to write, from 0 to ``num_slices - 1``.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Must be `None` or the memory configuration of `cache_tensor`. Defaults to `None`.
            output_dtype (ttnn.DataType, optional): Must be `None` or the dtype of `cache_tensor`. Defaults to `None`.

        Returns:
            ttnn.Tensor: `cache_tensor`, with `input_tensor` written into slice `slice_index`.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, INT32, UINT32, UINT16, UINT8, BFLOAT8_B, BFLOAT4_B
                 - TILE

        Memory Support:
            - Input: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED in L1
            - Output (`cache_tensor`): interleaved DRAM and L1
        )doc";

    ttnn::bind_function<"sharded_to_interleaved_partial">(
        mod,
        doc,
        &ttnn::sharded_to_interleaved_partial,
        nb::arg("input_tensor").noconvert(),
        nb::arg("cache_tensor").noconvert(),
        nb::arg("num_slices"),
        nb::arg("slice_index"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_dtype") = nb::none());
}

}  // namespace ttnn::operations::data_movement
