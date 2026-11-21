// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reshard_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "reshard.hpp"
#include "ttnn/types.hpp"
namespace ttnn::operations::data_movement {

void bind_reshard(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Copies the sharded tensor `input_tensor` into the sharded memory configuration `output_memory_config`, which
        can change the shard shape, the core grid, the memory layout and the buffer type. The output has the input's
        shape, dtype, layout and values.

        Args:
            input_tensor (ttnn.Tensor): the sharded input tensor, on device, see Memory Support. A TILE input must have 32x32 tiles.
            output_memory_config (ttnn.MemoryConfig): sharded memory configuration of the output, see Memory Support. Its shards must cover the input's shape: HEIGHT_SHARDED shards span the full width, WIDTH_SHARDED shards span the full height, and the grid has a core for each shard (for BLOCK_SHARDED with ROW_MAJOR orientation, one rectangle of cores with a row for each row of shards and a column for each column of shards). For a TILE tensor, the shard height and width must be multiples of 32. For a ROW_MAJOR WIDTH_SHARDED or BLOCK_SHARDED tensor, the input and output shard widths in bytes must be multiples of 16.
            output_tensor (ttnn.Tensor, optional): Tensor to write the output into, on the input's device. It must have the input's shape, dtype and layout and `output_memory_config` as its memory configuration. Defaults to `None`.

        Returns:
            ttnn.Tensor: the resharded tensor, with the memory configuration `output_memory_config`.

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
            - Input: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED in L1; HEIGHT_SHARDED and WIDTH_SHARDED in DRAM
            - Output: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED in L1; HEIGHT_SHARDED and WIDTH_SHARDED in DRAM
            - ROW_MAJOR: the input and output must have the same memory layout, and both be HEIGHT_SHARDED when either is in DRAM
        )doc";

    ttnn::bind_function<"reshard">(
        mod,
        doc,
        &ttnn::reshard,
        nb::arg("input_tensor").noconvert(),
        nb::arg("output_memory_config"),
        nb::arg("output_tensor").noconvert() = nb::none());
}

}  // namespace ttnn::operations::data_movement
