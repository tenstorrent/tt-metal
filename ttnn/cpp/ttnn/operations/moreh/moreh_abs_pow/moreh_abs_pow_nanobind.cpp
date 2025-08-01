// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_abs_pow_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "moreh_abs_pow.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace ttnn::operations::moreh::moreh_abs_pow {

void bind_moreh_abs_pow_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::moreh_abs_pow,
        "Moreh Pow Operation",
        ttnn::nanobind_arguments_t{
            nb::arg("input"),
            nb::arg("p"),
            nb::kw_only(),
            nb::arg("output") = std::nullopt,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("compute_kernel_config") = std::nullopt,
        });
}
}  // namespace ttnn::operations::moreh::moreh_abs_pow
