// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_sum_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/moreh/moreh_sum/moreh_sum.hpp"

namespace ttnn::operations::moreh::moreh_sum {
void bind_moreh_sum_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::moreh_sum,
        "Moreh Sum Operation",
        ttnn::nanobind_arguments_t{
            nb::arg("input"),
            nb::arg("dim"),
            nb::kw_only(),
            nb::arg("keepdim") = false,
            nb::arg("output") = std::nullopt,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("compute_kernel_config") = std::nullopt});
}
}  // namespace ttnn::operations::moreh::moreh_sum
