// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "non_zero_indices_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "non_zero_indices.hpp"

namespace ttnn::operations::data_movement {
namespace {

void bind_non_zero(nb::module_& mod) {
    const auto* doc = R"doc(
        Finds the non-zero elements of a rank-4 input tensor and returns their count and indices.

        Args:
            input_tensor (ttnn.Tensor): the input tensor. It must have rank 4.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration of the output tensors. It must be interleaved. Defaults to `None`, which uses DRAM.

        Returns:
            List[ttnn.Tensor]: two UINT32 ROW_MAJOR tensors. The first, of shape ``[1, 1, 1, 8]``, holds the number of non-zero elements N at index ``[0, 0, 0, 0]``. The second, of shape ``[1, 1, 1, 4 * volume]``, starts with the ``(n, c, h, w)`` index of each non-zero element in row-major order, so ``indices[0, 0, 0, :4 * N].reshape(N, 4)`` holds one ``(n, c, h, w)`` index per row. The values after those, and the other seven values of the first tensor, are not initialized.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, INT32, UINT32
                 - TILE, ROW_MAJOR

            The op runs on a single core, so the input volume is limited by the L1 size of one core.

        Memory Support:
            - Input: interleaved DRAM and L1; HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED in L1. A ROW_MAJOR input must not have shape padding, so a WIDTH_SHARDED or BLOCK_SHARDED ROW_MAJOR input needs a shard width that divides the last dimension
            - Output: interleaved DRAM or L1
    )doc";

    ttnn::bind_function<"nonzero">(
        mod,
        doc,
        &ttnn::nonzero,
        nb::arg("input_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());
}

}  // namespace

void bind_non_zero_indices(nb::module_& mod) { bind_non_zero(mod); }

}  // namespace ttnn::operations::data_movement
