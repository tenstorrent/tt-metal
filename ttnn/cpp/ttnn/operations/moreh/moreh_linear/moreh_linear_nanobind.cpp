// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_linear_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/moreh/moreh_linear/moreh_linear.hpp"

namespace nb = nanobind;

namespace ttnn::operations::moreh::moreh_linear {
void bind_moreh_linear_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::moreh_linear,
        "Moreh Linear Operation",
        ttnn::nanobind_arguments_t{
            nb::arg("input"),
            nb::arg("weight"),
            nb::kw_only(),
            nb::arg("bias") = std::nullopt,
            nb::arg("output") = std::nullopt,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("compute_kernel_config") = std::nullopt});
}
}  // namespace ttnn::operations::moreh::moreh_linear
