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

        Example:

        .. code-block:: python

            input_tensor = ttnn.from_torch(torch.rand((1, 1, 256), dtype=torch.bfloat16), device=device)
            output_tensor = ttnn.unsqueeze(input_tensor, 2)  # shape [1, 1, 1, 256]

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            dim (int): the index of the new dimension in the output, in ``[-r - 1, r]`` for an input of rank ``r``. Negative values count from the last dimension of the output.

        Returns:
            ttnn.Tensor: the input with one more dimension, with the same dtype and layout. A view keeps the input's memory layout, buffer type and shard grid, with shard metadata adapted to the new shape as in :func:`ttnn.view`. When data is copied, the output keeps the input's buffer type and normally its memory layout: a sharded output gets a shard spec derived for the new shape, as in :func:`ttnn.reshape`, and falls back to interleaved when the input's shard grid is not rectangular.

        Note:
            The unsqueeze is done with :func:`ttnn.reshape`. For a host tensor it is always a metadata-only view that
            shares the input's data. For a device tensor it is a zero-cost view that shares the input's buffer
            when the size of the last dimension does not change and, for TILE layout, the size of the second-to-last
            dimension does not change or is a multiple of 32 both before and after the unsqueeze. Otherwise the data
            is copied into a new tensor: for example, unsqueezing ``[32, 4]`` at ``-1`` copies, while unsqueezing
            ``[32, 1]`` at ``-1`` is a view in ROW_MAJOR layout but copies in TILE layout, where the
            second-to-last dimension goes from 32 to 1. An ND sharded input is copied when the unsqueeze changes its
            padded shape, as adding a dimension normally does. The exception is a rank 1 TILE input ``[W]``, whose
            padded shape is already rank 2, ``[32, W]`` with ``W`` rounded up to a multiple of 32: unsqueezing it at ``0`` to
            ``[1, W]`` keeps the padded shape, so the result is a view.

            With a tile other than 32x32, a view also has the padding and buffer-size limits of :func:`ttnn.view`: a
            view whose padded output does not fit in the input's buffer raises an error instead of copying, as for
            ``[16, 32]`` with 16x32 tiles unsqueezed at ``0``.

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
            - Sharded: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED, in DRAM or L1. A DRAM-sharded ROW_MAJOR, BFLOAT8_B or BFLOAT4_B input is supported only when the unsqueeze is a view. An L1-sharded ROW_MAJOR copy needs input and output shard rows whose size in bytes is a multiple of the L1 alignment, unless the output falls back to interleaved; a ROW_MAJOR copy adds a last dimension of size 1, so the derived output shards are one element wide, and it works only when the output falls back to interleaved, as for a non-rectangular shard grid
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
