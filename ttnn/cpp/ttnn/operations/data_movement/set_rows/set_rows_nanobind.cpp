// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "set_rows_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "set_rows.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_set_rows_operation(nb::module_& mod) {
    const auto* doc = R"doc(
        Writes rows of an updates tensor into an input tensor on device, leaving every other row
        untouched: ``output[indices[i]] = updates[i]``. Equivalent to
        ``torch.index_put_(input, (indices,), updates)`` for non-negative in-range indices.

        Indices must be non-negative and smaller than the number of input rows; out-of-range
        indices are rejected before dispatch (matching ``torch.index_put_``, which raises). With
        duplicate indices the later update wins (``index_copy_`` order).

        Implemented on top of ``ttnn.indexed_fill``; memory-configuration support is inherited
        from it.

        Args:
            input (ttnn.Tensor): The [H, W] row-major destination tensor whose rows are replaced.
            indices (ttnn.Tensor): The [N] row-major uint32/int32 vector of destination row indices.
            updates (ttnn.Tensor): The [N, W] row-major source rows, same dtype and width as the input.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Specifies the memory configuration for the output tensor. Defaults to `None`.
            out (ttnn.Tensor, optional): A preallocated tensor to store the result. Defaults to `None`.

        Returns:
            ttnn.Tensor: the [H, W] result (the input with the selected rows replaced).

        Note:

            Supported dtypes and layouts for this milestone:

            .. list-table::
                :header-rows: 1

                * - Tensor
                  - Dtypes
                  - Layouts
                * - input / updates
                  - matching dtypes (BFLOAT16 primary; others inherited from ``indexed_fill``)
                  - ROW_MAJOR
                * - indices
                  - UINT32, INT32
                  - ROW_MAJOR
    )doc";

    ttnn::bind_function<"set_rows">(
        mod,
        doc,
        &ttnn::set_rows,
        nb::arg("input").noconvert(),
        nb::arg("indices"),
        nb::arg("updates"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("out") = nb::none());
}

}  // namespace ttnn::operations::data_movement::detail
