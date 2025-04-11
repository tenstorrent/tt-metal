// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_mean_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/moreh/moreh_mean/moreh_mean.hpp"

namespace ttnn::operations::moreh::moreh_mean {
void bind_moreh_mean_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::moreh_mean,
        "Moreh Mean Operation",
        ttnn::nanobind_arguments_t{
            nb::arg("input"),
            nb::kw_only(),
            nb::arg("dim"),
            nb::arg("keepdim") = false,
            nb::arg("divisor") = nb::none(),
            nb::arg("output") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()});
}
}  // namespace ttnn::operations::moreh::moreh_mean
