// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "interleaved_to_sharded_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/array.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "interleaved_to_sharded.hpp"
#include "ttnn/types.hpp"
namespace ttnn::operations::data_movement {

void bind_interleaved_to_sharded(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Copies an interleaved tensor into a sharded tensor with the memory configuration `sharded_memory_config`. The
        output has the input's shape, layout and values. `output_dtype` converts a TILE input to another dtype.

        A second overload builds an L1 sharded memory configuration from a core grid: ``ttnn.interleaved_to_sharded(input_tensor, grid, shard_shape, shard_scheme, shard_orientation, output_dtype=None, *, keep_l1_aligned=False)``.
        ``grid`` is a ``ttnn.CoreRangeSet`` of the cores to use, or a ``ttnn.CoreCoord`` grid size from which one core
        per shard is taken, a row of the grid at a time for ``ROW_MAJOR`` orientation or a column at a time for
        ``COL_MAJOR``. For ``BLOCK_SHARDED`` with a ``ttnn.CoreCoord`` grid, the grid's x size (``ROW_MAJOR``) or y size
        (``COL_MAJOR``) must equal the number of shards across the width. ``shard_shape`` is the ``[height, width]`` of
        each shard, ``shard_scheme`` is ``ttnn.TensorMemoryLayout.HEIGHT_SHARDED``, ``WIDTH_SHARDED`` or
        ``BLOCK_SHARDED``, and ``shard_orientation`` is a ``ttnn.ShardOrientation``.

        Args:
            input_tensor (ttnn.Tensor): the interleaved input tensor, on device. A TILE input must have 32x32 tiles.
            sharded_memory_config (ttnn.MemoryConfig): sharded memory configuration of the output, see Memory Support. For a TILE input, the shard height and width must be multiples of 32. For a ROW_MAJOR input, the shard width in bytes must be a multiple of 16.
            output_dtype (ttnn.DataType, optional): Data type of the output. Defaults to `None`, which keeps the input dtype. A TILE input of BFLOAT16, FLOAT32 or BFLOAT8_B can be converted to any of these three dtypes. Input values must not be NaN when the dtype changes.

        Keyword Args:
            keep_l1_aligned (bool, optional): Defaults to `False`.
            preallocated_output (ttnn.Tensor, optional): Tensor to write the output into, on the input's device. It must have the input's shape and layout, the output dtype and `sharded_memory_config` as its memory configuration. Defaults to `None`.

        Returns:
            ttnn.Tensor: the sharded output tensor, with the input's shape and layout.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, INT32, UINT32, UINT16, UINT8
                 - TILE, ROW_MAJOR
               * - BFLOAT8_B, BFLOAT4_B
                 - TILE

            On Blackhole, the input can also be an ``FP8_E4M3`` tensor in ROW_MAJOR layout.

        Memory Support:
            - Input: interleaved DRAM and L1
            - Output: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED in L1; HEIGHT_SHARDED and WIDTH_SHARDED in DRAM
            - ND sharded output: the ND shard spec must be equivalent to a HEIGHT_SHARDED, WIDTH_SHARDED or BLOCK_SHARDED shard spec
        )doc";

    ttnn::bind_function<"interleaved_to_sharded">(
        mod,
        doc,

        // Overload 1: Using grid, shard_shape, shard_scheme, shard_orientation (detailed)
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                const std::variant<CoreCoord, CoreRangeSet>&,
                std::array<uint32_t, 2>,
                TensorMemoryLayout,
                tt::tt_metal::ShardOrientation,
                const std::optional<ttnn::DataType>&,
                const std::optional<bool>&>(&ttnn::interleaved_to_sharded),
            nb::arg("input_tensor").noconvert(),
            nb::arg("grid"),
            nb::arg("shard_shape"),
            nb::arg("shard_scheme"),
            nb::arg("shard_orientation"),
            nb::arg("output_dtype") = nb::none(),
            nb::kw_only(),
            nb::arg("keep_l1_aligned") = false),

        // Overload 2: Using MemoryConfig (simple)
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                const MemoryConfig&,
                const std::optional<ttnn::DataType>&,
                const std::optional<bool>&,
                const std::optional<ttnn::Tensor>&>(&ttnn::interleaved_to_sharded),
            nb::arg("input_tensor").noconvert(),
            nb::arg("sharded_memory_config"),
            nb::arg("output_dtype") = nb::none(),
            nb::kw_only(),
            nb::arg("keep_l1_aligned") = false,
            nb::arg("preallocated_output") = nb::none()));
}

}  // namespace ttnn::operations::data_movement
