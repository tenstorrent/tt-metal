// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/quasar/unary/unary.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar::detail {

void bind_unary(nb::module_& mod) {
    const auto* cos_doc =
        R"doc(
        Applies cos to :attr:`input_tensor` element-wise (Quasar / Metal 2.0).

        .. math::
            \mathrm{{output\_tensor}}_i = \verb|cos|(\mathrm{{input\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`
                (the input's memory configuration).
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32
                 - TILE
        )doc";

    ttnn::bind_function<"cos", "ttnn.experimental.quasar.">(
        mod,
        cos_doc,
        &ttnn::operations::experimental::quasar::cos,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

}  // namespace ttnn::operations::experimental::quasar::detail
