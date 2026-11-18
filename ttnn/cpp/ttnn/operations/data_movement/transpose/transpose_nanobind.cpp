// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "transpose_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"

#include "transpose.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_transpose(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Returns a tensor with dimensions ``dim1`` and ``dim2`` of the input tensor swapped.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.
                dim1 (int): the first dimension to swap. Negative values count from the last dimension.
                dim2 (int): the second dimension to swap. Negative values count from the last dimension.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`: an interleaved input keeps its memory configuration, and an L1 sharded input gives an L1 output, sharded or interleaved depending on the input's shard spec and on the dimensions swapped. A given ``memory_config`` sets the output's memory configuration.
                pad_value (float, optional): value that fills the tile padding that the transpose exposes on a ``BFLOAT16``, ``FLOAT32`` or ``UINT32`` TILE output when exactly one of ``dim1`` and ``dim2`` is among the last two dimensions. Must be `0.0` when ``dim1`` and ``dim2`` are both among the last two dimensions or both outside them. Defaults to `0.0`.

            Returns:
                ttnn.Tensor: the output tensor, with the same dtype and layout as the input.

            Note:
                Supported dtypes and layouts:

                .. list-table::
                   :header-rows: 1

                   * - Dtypes
                     - Layouts
                   * - BFLOAT16, FLOAT32, INT32, UINT32
                     - TILE, ROW_MAJOR
                   * - UINT8
                     - ROW_MAJOR
                   * - BFLOAT8_B
                     - TILE

                ``BFLOAT8_B`` is supported for interleaved tensors of rank 4 or less.

                ``BFLOAT16`` input values must not be NaN when ``dim1`` or ``dim2`` is the last dimension.

            Memory Support:
                - Interleaved: DRAM and L1
                - Sharded: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED input and output in L1, for ``BFLOAT16`` and ``FLOAT32`` tensors
        )doc";

    ttnn::bind_function<"transpose">(
        mod,
        doc,

        // Overload 1: with memory_config
        ttnn::overload_t(
            nb::overload_cast<const ttnn::Tensor&, int64_t, int64_t, const std::optional<ttnn::MemoryConfig>&, float>(
                &ttnn::transpose),
            nb::arg("input_tensor"),
            nb::arg("dim1"),
            nb::arg("dim2"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("pad_value") = 0.0f),

        // Overload 2: without memory_config
        ttnn::overload_t(
            nb::overload_cast<const ttnn::Tensor&, int64_t, int64_t, float>(&ttnn::transpose),
            nb::arg("input_tensor"),
            nb::arg("dim1"),
            nb::arg("dim2"),
            nb::arg("pad_value") = 0.0f));
}
}  // namespace ttnn::operations::data_movement::detail
