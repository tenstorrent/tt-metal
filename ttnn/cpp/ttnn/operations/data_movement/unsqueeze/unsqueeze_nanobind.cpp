// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unsqueeze_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

void bind_unsqueeze(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Returns `input_tensor` with a new dimension of size 1 inserted at position `dim`, like ``torch.unsqueeze``.

        When the new dimension comes before the last two dimensions of the output (`dim` in ``[-rank - 1, -3]`` or ``[0, rank - 2]``), the output is a view of `input_tensor`: it shares the input's buffer and no data moves.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            dim (int): the position of the new dimension in the output, in ``[-rank - 1, rank]``, where ``rank`` is the rank of `input_tensor`. Negative values count from the end of the output, so ``-1`` adds a last dimension.

        Returns:
            ttnn.Tensor: the output tensor, with one more dimension than `input_tensor` and the input's dtype, layout and memory configuration.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, INT32, UINT32
                 - TILE, ROW_MAJOR
               * - BFLOAT8_B, BFLOAT4_B
                 - TILE

            Also supported when the new dimension comes before the last two dimensions of the output:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - UINT16, UINT8
                 - TILE, ROW_MAJOR

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED in L1, when the new dimension comes before the last two dimensions of the output
        )doc";

    ttnn::bind_function<"unsqueeze">(
        mod,
        doc,
        nb::overload_cast<const ttnn::Tensor&, int>(&ttnn::unsqueeze),
        nb::arg("input_tensor"),
        nb::arg("dim"));
}

}  // namespace ttnn::operations::data_movement
