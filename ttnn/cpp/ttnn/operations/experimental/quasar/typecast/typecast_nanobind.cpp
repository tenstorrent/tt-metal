// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "typecast.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar::detail {

void bind_typecast(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Changes the data type of the input tensor (Quasar / Metal 2.0).

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            dtype (ttnn.DataType): the target (output) data type.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            optional_output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            sub_core_grids (ttnn.CoreRangeSet, optional): restrict execution to a subset of cores. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            A second overload accepts an explicit ``(input_dtype, output_dtype)`` pair to reinterpret the input's
            declared dtype before casting.
        )doc";

    ttnn::bind_function<"typecast", "ttnn.experimental.quasar.">(
        mod,
        doc,
        // Overload 1: single (output) dtype -- mirrors ttnn.typecast(x, dtype).
        ttnn::overload_t(
            nb::overload_cast<
                const Tensor&,
                const DataType&,
                const std::optional<MemoryConfig>&,
                const std::optional<Tensor>&,
                const std::optional<CoreRangeSet>&>(&ttnn::operations::experimental::quasar::typecast),
            nb::arg("input_tensor").noconvert(),
            nb::arg("dtype").noconvert(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("optional_output_tensor") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()),
        // Overload 2: explicit (input_dtype, output_dtype).
        ttnn::overload_t(
            nb::overload_cast<
                const Tensor&,
                const DataType&,
                const DataType&,
                const std::optional<MemoryConfig>&,
                const std::optional<Tensor>&,
                const std::optional<CoreRangeSet>&>(&ttnn::operations::experimental::quasar::typecast),
            nb::arg("input_tensor").noconvert(),
            nb::arg("input_dtype").noconvert(),
            nb::arg("output_dtype").noconvert(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("optional_output_tensor") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()));
}

}  // namespace ttnn::operations::experimental::quasar::detail
