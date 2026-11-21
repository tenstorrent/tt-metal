// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "interleaved_to_sharded_partial_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "interleaved_to_sharded_partial.hpp"
#include "ttnn/types.hpp"
namespace ttnn::operations::data_movement {

void bind_interleaved_to_sharded_partial(nb::module_& mod) {
    const auto* doc = R"doc(
        Copies one slice of an interleaved TILE tensor into a sharded tensor in L1.

        The input of shape ``[N, C, H, W]`` is read as ``N*C*H`` rows of width ``W``. The rows are split into
        `num_slices` equal slices, and the output holds slice `slice_index`. ``ttnn.sharded_to_interleaved_partial`` with
        the same `num_slices` and `slice_index` writes the slice back into the same rows of an interleaved TILE tensor
        with the input's shape and the output's dtype.

        Args:
            input_tensor (ttnn.Tensor): the interleaved input tensor, on device, of shape ``[N, C, H, W]``, in TILE layout with 32x32 tiles. ``H`` and ``W`` must be multiples of 32.
            grid (ttnn.CoreCoord): size of the core grid to shard over. One core per shard is taken from it, a row of the grid at a time for ``ROW_MAJOR`` orientation or a column at a time for ``COL_MAJOR``. It must fit in the device's compute grid. For ``BLOCK_SHARDED``, its x size (``ROW_MAJOR``) or y size (``COL_MAJOR``) must equal the number of shards across the width.
            shard_shape (List[int]): ``[height, width]`` of each shard. Both must be positive multiples of 32. For ``HEIGHT_SHARDED`` the width must be ``W``, and for ``WIDTH_SHARDED`` the height must be ``N*C*H / num_slices``.
            num_slices (int): number of slices to split the rows into. Must be at least 1, and ``N*C*H / num_slices`` must be a multiple of 32.
            slice_index (int): index of the slice to copy, from 0 to ``num_slices - 1``.
            shard_scheme (ttnn.TensorMemoryLayout): ``HEIGHT_SHARDED``, ``WIDTH_SHARDED`` or ``BLOCK_SHARDED``.
            shard_orientation (ttnn.ShardOrientation): ``ROW_MAJOR`` or ``COL_MAJOR`` order of the shards on the cores.

        Keyword Args:
            output_dtype (ttnn.DataType, optional): Data type of the output. Defaults to `None`, which keeps the input dtype. An input of BFLOAT16, FLOAT32 or BFLOAT8_B can be converted to any of these three dtypes. Input values must not be NaN when the dtype changes.

        Returns:
            ttnn.Tensor: the slice, of shape ``[1, 1, N*C*H / num_slices, W]``, in TILE layout and sharded in L1 with `shard_scheme`.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, INT32, UINT32, UINT16, UINT8, BFLOAT8_B, BFLOAT4_B
                 - TILE

        Memory Support:
            - Input: interleaved DRAM and L1
            - Output: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED in L1
        )doc";

    ttnn::bind_function<"interleaved_to_sharded_partial">(
        mod,
        doc,
        nb::overload_cast<
            const ttnn::Tensor&,
            const std::variant<CoreCoord, CoreRangeSet>&,
            const std::array<uint32_t, 2>&,
            int64_t&,
            int64_t&,
            tt::tt_metal::TensorMemoryLayout,
            tt::tt_metal::ShardOrientation,
            const std::optional<ttnn::DataType>&>(&ttnn::interleaved_to_sharded_partial),
        nb::arg("input_tensor").noconvert(),
        nb::arg("grid"),
        nb::arg("shard_shape"),
        nb::arg("num_slices"),
        nb::arg("slice_index"),
        nb::arg("shard_scheme"),
        nb::arg("shard_orientation"),
        nb::kw_only(),
        nb::arg("output_dtype") = nb::none());
}

}  // namespace ttnn::operations::data_movement
