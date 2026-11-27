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
        Returns a tensor with a new dimension of size 1 inserted at the specified position.

        Equivalent pytorch code:

        .. code-block:: python

            input_tensor = torch.rand((1,1,256), dtype=torch.bfloat16)
            output_tensor = torch.unsqueeze(input_tensor, 2) # tensor of shape (1,1,1,256), where at dimension 2 we added a new dim of size 1

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            dim (int): the index of the new dimension in the output, in ``[-r - 1, r]`` for an input of rank ``r``. Negative values count from the last dimension of the output.

        Returns:
            ttnn.Tensor: the input with one more dimension, with the same dtype, layout and memory configuration.

        Note:
            The unsqueeze is done with :func:`ttnn.reshape`. It is a zero-cost view that shares the input's buffer
            when the last dimension is kept and, for TILE layout, the second-to-last dimension does not change or is a
            multiple of 32 both before and after the unsqueeze. Otherwise, as when ``dim`` adds a new last dimension
            (``dim`` equal to ``r`` or ``-1``), the data is copied into a new tensor. An ND sharded input is always
            copied, since adding a dimension changes its padded shape.

            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, INT32, UINT32, UINT16, UINT8
                 - TILE, ROW_MAJOR
               * - BFLOAT8_B, BFLOAT4_B
                 - TILE

            An unsqueeze that copies data has the dtype restrictions of :func:`ttnn.reshape`.

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED
            - ND sharded: supported as in :func:`ttnn.reshape`, through a DRAM-interleaved intermediate when data is copied
        )doc";

    ttnn::bind_function<"unsqueeze">(
        mod,
        doc,
        nb::overload_cast<const ttnn::Tensor&, int>(&ttnn::unsqueeze),
        nb::arg("input_tensor"),
        nb::arg("dim"));
}

}  // namespace ttnn::operations::data_movement
