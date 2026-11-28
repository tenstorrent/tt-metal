// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "get_rows_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "get_rows.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_get_rows_operation(nb::module_& mod) {
    const auto* doc = R"doc(
        Gathers rows of a row-major input tensor on device, preserving the requested index order:
        ``output[i] = input[indices[i]]``. Equivalent to ``torch.index_select(input, 0, indices)``.

        Indices must be non-negative and smaller than the number of input rows; out-of-range
        indices are rejected before dispatch (matching ``torch.index_select``, which raises).

        Args:
            input (ttnn.Tensor): The [H, W] row-major source tensor whose rows are gathered.
            indices (ttnn.Tensor): The [N] row-major uint32/int32 vector of row indices to gather.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Specifies the memory configuration for the output tensor. Defaults to `None`.
            out (ttnn.Tensor, optional): A preallocated tensor to store the gathered rows. Defaults to `None`.

        Returns:
            ttnn.Tensor: the [N, W] gathered rows, in index order.

        Note:

            Supported dtypes and layouts for this milestone:

            .. list-table::
                :header-rows: 1

                * - Tensor
                  - Dtypes
                  - Layouts
                * - input
                  - BFLOAT16, FLOAT32, UINT32, INT32
                  - ROW_MAJOR
                * - indices
                  - UINT32, INT32
                  - ROW_MAJOR

            Both tensors must use interleaved memory (DRAM or L1); sharded inputs, indices and
            outputs are rejected in this milestone.
    )doc";

    ttnn::bind_function<"get_rows">(
        mod,
        doc,
        &ttnn::get_rows,
        nb::arg("input").noconvert(),
        nb::arg("indices"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("out") = nb::none());
}

}  // namespace ttnn::operations::data_movement::detail
