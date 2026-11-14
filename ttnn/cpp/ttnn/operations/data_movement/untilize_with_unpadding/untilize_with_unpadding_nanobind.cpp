// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "untilize_with_unpadding.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_untilize_with_unpadding(nb::module_& mod) {
    // The BFLOAT16 NaN rule in this docstring tracks tenstorrent/tt-metal#31406; drop it once that issue is fixed.
    const auto* doc =
        R"doc(
        Changes the layout of the input tensor from TILE to ROW_MAJOR, keeping indices 0 through ``output_tensor_end[i]`` of
        each dimension ``i``.

        Args:
            input_tensor (ttnn.Tensor): the input tensor, on device.
            output_tensor_end (ttnn.Shape or List[int]): the last index to keep in each dimension. The end is inclusive, so dimension ``i`` of the output has size ``output_tensor_end[i] + 1``. It has one entry for each dimension of the input. Each entry must be less than the input's size in that dimension, or ``2**32 - 1``, which gives the output a size of 0 in that dimension; a dimension of size 0 takes ``2**32 - 1``. The kept part of the dimensions before the last two must be contiguous: once one of them keeps more than one index, each later one must keep all of its indices.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration of the output. Defaults to `None`, which uses the input tensor's memory configuration. The operation can derive a sharded output's shard shape from the output shape instead of taking it from ``memory_config``, and a HEIGHT_SHARDED, WIDTH_SHARDED or BLOCK_SHARDED input whose memory layout matches the output's keeps its own core grid.
            use_multicore (bool, optional): Whether to use multicore. Defaults to `True`.
            sub_core_grids (ttnn.CoreRangeSet, optional): Cores to run the operation on. Defaults to `None`, which lets the operation choose its cores. It needs an interleaved input.

        Returns:
            ttnn.Tensor: the output tensor, in ROW_MAJOR layout with the input's dtype. A ``BFLOAT8_B`` input gives a ``BFLOAT16`` output.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, BFLOAT8_B, INT32, UINT32, UINT16, UINT8
                 - TILE

            ``BFLOAT16`` input values must not be NaN.

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded input: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED in L1, with an interleaved output or an L1 output of the same memory layout
            - Sharded output from an interleaved input: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED in L1; HEIGHT_SHARDED and WIDTH_SHARDED in DRAM
            - A BLOCK_SHARDED input must hold one matrix: every dimension before the last two has size 1
            - A WIDTH_SHARDED input keeps every dimension before the last two. With an interleaved output it must hold one matrix, and its padded width minus the output width must be less than its shard width. With a WIDTH_SHARDED output, the output width must equal the input's padded width or need every input shard: ``ceil(output_width / shard_width)`` equals the number of input cores
            - A HEIGHT_SHARDED output from a HEIGHT_SHARDED input that holds more than one matrix keeps every row
            - ND sharded: L1 input, with an interleaved or sharded output; an ND sharded output needs an ND sharded input
    )doc";

    ttnn::bind_function<"untilize_with_unpadding">(
        mod,
        doc,
        &ttnn::untilize_with_unpadding,
        nb::arg("input_tensor"),
        nb::arg("output_tensor_end"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("use_multicore") = true,
        nb::arg("sub_core_grids") = nb::none());
}

}  // namespace ttnn::operations::data_movement::detail
