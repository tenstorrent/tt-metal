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
    const auto* doc =
        R"doc(
        Changes the layout of the input tensor from TILE to ROW_MAJOR and removes the elements past `output_tensor_end`.

        Args:
            input_tensor (ttnn.Tensor): the input tensor, on device.
            output_tensor_end (ttnn.Shape or List[int]): the last index to keep in each dimension. The end is inclusive, so the output size of dimension ``i`` is ``output_tensor_end[i] + 1``. It must have the input's rank and stay within the input's padded shape.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`, which uses the input tensor's memory configuration.
            use_multicore (bool, optional): Whether to use multicore. Defaults to `True`. Sharded outputs always use multicore.
            sub_core_grids (ttnn.CoreRangeSet, optional): Cores to run the operation on. Defaults to `None`, which uses the whole device. Not supported with a sharded input.

        Returns:
            ttnn.Tensor: the output tensor, in ROW_MAJOR layout with the input's dtype. ``BFLOAT8_B`` input gives ``BFLOAT16`` output.

        Note:
            Supported dtypes and layouts for the input tensor:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, BFLOAT8_B, INT32, UINT32, UINT16, UINT8
                 - TILE

        Memory Support:
            - Interleaved: DRAM and L1 input; the output can be interleaved, HEIGHT_SHARDED, WIDTH_SHARDED or BLOCK_SHARDED
            - Sharded: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED input in L1; the output can be interleaved or sharded
            - ND sharded: input and output; an ND sharded output needs an ND sharded input
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
