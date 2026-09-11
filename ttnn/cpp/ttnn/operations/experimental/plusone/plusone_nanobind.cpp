// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "plusone_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/plusone/plusone.hpp"

namespace ttnn::operations::experimental::plusone::detail {
void bind_experimental_plusone_operation(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Returns input tensor elements increased by 1.
            Input tensor must have INT32 or UINT32 data type, ROW_MAJOR layout, and 1 to 4 dimensions.
            This op only gives decent performance for small tensors (up to 100 elements).

            Elementwise behaviour:

            * ``skip_negative_entries = False`` (default): every element is incremented, ``output[i] = input[i] + 1``.
            * ``skip_negative_entries = True``: only elements in ``[0, INT32_MAX)`` are incremented; negative
              elements, and ``INT32_MAX`` (which would overflow to a negative value), are returned unchanged.
              This preserves negative sentinel values, e.g. ``-1`` marking an inactive user slot in a decode
              position tensor.

            This op also allows you to specify the core to use in the sub_core_grids argument.
            If the input tensor is L1 sharded on the sub core grid, each individual shard will be incremented with output residing in L1 of same sub core grid.
            If the input tensor is DRAM interleaved, only 1 core should be used as the sub core grid (uses 1 core by default).

            Args:
                * :attr:`input_tensor`: Input Tensor for plusone.
                * :attr:`sub_core_grids`: Sub core grid of cores where the addition would take place
                * :attr:`skip_negative_entries`: bool flag to skip incrementing values that are negative or overflow past INT32_MAX. Defaults to False

        )doc";

    ttnn::bind_function<"plus_one">(
        mod,
        doc,
        &ttnn::operations::experimental::plus_one,
        nb::arg("input_tensor").noconvert(),
        nb::arg("sub_core_grids") = nb::none(),
        nb::arg("skip_negative_entries") = false);
}

}  // namespace ttnn::operations::experimental::plusone::detail
