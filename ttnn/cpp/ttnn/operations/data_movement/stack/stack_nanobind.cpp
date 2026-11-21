// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "stack_nanobind.hpp"

#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/data_movement/stack/stack.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

void bind_stack(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Stacks `input_tensors` along a new dimension `dim`. The output has one more dimension than the inputs, of size
        ``len(input_tensors)`` at position `dim`.

        Args:
            input_tensors (List[ttnn.Tensor]): the tensors to stack, on the same device. Must hold at least one tensor. The tensors must have the same shape, dtype and layout, and at least one element.
            dim (int): the position of the new dimension in the output, in ``[-(r + 1), r]`` for inputs of rank ``r``. Negative values count from the last dimension of the output, so -1 and ``r`` both add a new last dimension.

        Returns:
            ttnn.Tensor: the stacked tensor, of rank ``r + 1``, interleaved in DRAM, with the dtype and layout of the inputs.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, INT32, UINT32
                 - TILE, ROW_MAJOR
               * - UINT16, UINT8
                 - ROW_MAJOR
               * - BFLOAT8_B
                 - TILE

            ``BFLOAT8_B`` tensors need the new dimension to come before the last two dimensions of the output: `dim`
            from 0 to ``r - 2``, or from ``-(r + 1)`` to -3.

            ``BFLOAT16`` input values must not be NaN when the new dimension is the last dimension of the output
            (`dim` is ``r`` or -1), or, for TILE inputs, the second-to-last (`dim` is ``r - 1`` or -2).

        Memory Support:
            - Input: interleaved DRAM and L1
            - Output: interleaved DRAM
        )doc";

    ttnn::bind_function<"stack">(mod, doc, &ttnn::stack, nb::arg("input_tensors"), nb::arg("dim"));
}

}  // namespace ttnn::operations::data_movement
