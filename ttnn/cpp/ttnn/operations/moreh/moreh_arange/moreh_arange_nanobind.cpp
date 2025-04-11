// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_arange_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "moreh_arange.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace ttnn::operations::moreh::moreh_arange {
void bind_moreh_arange_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::moreh_arange,
        "Moreh Arange Operation",
        ttnn::nanobind_arguments_t{
            nb::arg("start") = 0,
            nb::arg("end"),
            nb::arg("step") = 1,
            nb::arg("any"),
            nb::kw_only(),
            nb::arg("output") = std::nullopt,
            nb::arg("untilize_out") = false,
            nb::arg("dtype") = std::nullopt,
            nb::arg("memory_config") = std::nullopt,
        });
}
}  // namespace ttnn::operations::moreh::moreh_arange
