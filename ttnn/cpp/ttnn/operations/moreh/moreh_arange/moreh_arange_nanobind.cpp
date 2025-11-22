// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
            nb::arg("output") = nb::none(),
            nb::arg("untilize_out") = false,
            nb::arg("dtype") = nb::none(),
            nb::arg("memory_config") = nb::none(),
        });
}
}  // namespace ttnn::operations::moreh::moreh_arange
