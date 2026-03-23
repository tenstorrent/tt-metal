// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_abs_pow_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "moreh_abs_pow.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::moreh::moreh_abs_pow {

void bind_moreh_abs_pow_operation(nb::module_& mod) {
    ttnn::bind_function<"moreh_abs_pow">(
        mod,
        "Moreh Pow Operation",
        ttnn::overload_t(
            &ttnn::moreh_abs_pow,
            nb::arg("input"),
            nb::arg("p"),
            nb::kw_only(),
            nb::arg("output") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()));
}
}  // namespace ttnn::operations::moreh::moreh_abs_pow
