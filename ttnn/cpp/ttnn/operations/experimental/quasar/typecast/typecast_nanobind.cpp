// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "typecast.hpp"

namespace ttnn::operations::experimental::quasar::detail {

void bind_typecast(nb::module_& mod) {
    const auto* doc = R"doc(
        Returns a tensor containing the input values converted to ``dtype``.

        The output preserves the input layout and uses ``memory_config`` when provided.
    )doc";

    ttnn::bind_function<"typecast", "ttnn.experimental.quasar.">(
        mod,
        doc,
        nb::overload_cast<
            const Tensor&,
            const DataType&,
            const std::optional<MemoryConfig>&,
            const std::optional<Tensor>&,
            const std::optional<CoreRangeSet>&>(&ttnn::operations::experimental::quasar::typecast),
        nb::arg("input_tensor").noconvert(),
        nb::arg("dtype").noconvert(),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("optional_output_tensor") = nb::none(),
        nb::arg("sub_core_grids") = nb::none());
}

}  // namespace ttnn::operations::experimental::quasar::detail
