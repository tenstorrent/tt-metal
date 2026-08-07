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
        No data is copied or moved.

        Note:
            * Input tensor must be stored on the device.
            * Currently supports only DRAM INTERLEAVED or L1 sharded tensors.
            * For DRAM INTERLEAVED tensors, narrow can only be performed on the first non-trivial dimension,
              with ``start`` pointing to the first DRAM bank.
            * For L1 sharded tensors, narrow is supported only when the narrowed region consists of complete
              full shards, or spans multiple shards with the same page offset.
            * For TILE_LAYOUT, ``start`` and ``length`` on the height or width dimension must be multiples of 32.
            * Negative values for ``dim`` and ``start`` are supported.

        Args:
            * input_tensor (ttnn.Tensor): Input tensor. Must be on device.
            * dim (int): Dimension along which to narrow. Supports negative indexing.
            * start (int): Starting index (inclusive). Supports negative indexing.
            * length (int): Length of the narrowed dimension. Must be > 0.

        Returns:
            ttnn.Tensor: A view of the input tensor with ``shape[dim] == length``.

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
