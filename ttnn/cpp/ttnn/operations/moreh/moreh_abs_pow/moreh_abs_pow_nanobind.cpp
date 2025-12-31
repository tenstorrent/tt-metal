// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_abs_pow_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "moreh_abs_pow.hpp"

namespace nb = nanobind;

namespace ttnn::operations::moreh::moreh_abs_pow {

void bind_moreh_abs_pow_operation(nb::module_& mod) {
    mod.def(
        "moreh_abs_pow",
        &ttnn::moreh_abs_pow,
        R"doc(Moreh Pow Operation)doc",
        nb::arg("input"),
        nb::arg("p"),
        nb::kw_only(),
        nb::arg("output") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}
}  // namespace ttnn::operations::moreh::moreh_abs_pow
