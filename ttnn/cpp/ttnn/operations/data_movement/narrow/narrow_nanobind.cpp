// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "narrow_nanobind.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/nanobind.h>

#include "narrow.hpp"

namespace ttnn::operations::data_movement {

void bind_narrow(nb::module_& mod) {
    const auto* doc = R"doc(
        Returns a narrowed view of the input tensor along dimension :attr:`dim`, starting at index :attr:`start`
        with the given :attr:`length`. Equivalent to `torch.narrow`.

        This is a zero-cost operation: the returned tensor shares the same data buffer as the input tensor.
        No data is copied or moved, so writes through either tensor are visible in the other.

        Args:
            input_tensor (ttnn.Tensor): the input tensor, on device. A multi-device tensor must be replicated.
            dim (int): the dimension to narrow, in ``[-r, r)`` for an input of rank ``r``. Negative values count from the last dimension.
            start (int): the starting index (inclusive), in ``[-shape[dim], shape[dim])``. Negative values count from the end of ``dim``.
            length (int): the length of the narrowed dimension. Must be greater than 0, with ``start + length <= shape[dim]``.

        Returns:
            ttnn.Tensor: a view of the input tensor with ``shape[dim] == length``, and the input's dtype, layout, buffer type and memory layout. For a sharded input, the shard spec is narrowed to the shards that hold the window. If the window covers all of ``dim``, the input tensor itself is returned.

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

            ``start`` and ``length`` are checked against the logical shape, not the tile-padded shape, so the window
            must lie within the logical size of ``dim``, as in `torch.narrow`.

            For a TILE input narrowed on one of its last two dims, ``start`` and ``length`` must be multiples of 32.
            For a ROW_MAJOR input narrowed on its last dim, ``start``, counted in elements, must be a multiple of the
            buffer's alignment in bytes, such as 16 for L1 on Wormhole.

            A DRAM interleaved input can only be narrowed on its first dimension of size greater than 1: every
            dimension before ``dim``, in the tile-padded shape, must have size 1. So the last dim cannot be narrowed,
            and a TILE input can be narrowed on its second-to-last dim only when all leading dims are 1. ``start``
            must also fall on a page held by the first DRAM bank, that is, the page index of ``start`` must be a
            multiple of the number of DRAM banks.

            An L1 sharded input can be narrowed only when the window consists of complete shards, or fits inside a
            shard along ``dim`` and spans shards at the same page offset. Narrowing the last dim inside a shard also
            needs shards that are one page tall: 32 rows for TILE, 1 row for ROW_MAJOR.

        Memory Support:
            - Interleaved: DRAM; L1 interleaved inputs are not supported
            - Sharded: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED in L1; DRAM-sharded inputs are not supported

        Example:

            >>> tensor = ttnn.rand((32, 16, 16, 4), dtype=ttnn.bfloat16, device=device)
            >>> output = ttnn.narrow(tensor, 0, 0, 12)
            >>> print(output.shape)
            ttnn.Shape([12, 16, 16, 4])

        )doc";

    ttnn::bind_function<"narrow">(
        mod, doc, &ttnn::narrow, nb::arg("input_tensor"), nb::arg("dim"), nb::arg("start"), nb::arg("length"));
}
}  // namespace ttnn::operations::data_movement
