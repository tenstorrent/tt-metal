// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_cumsum_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/moreh/moreh_cumsum/moreh_cumsum.hpp"

namespace ttnn::operations::moreh::moreh_cumsum {

void bind_moreh_cumsum_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::moreh_cumsum,
        "Moreh Cumsum Operation",
        ttnn::nanobind_arguments_t{
            nb::arg("input"),
            nb::arg("dim"),
            nb::kw_only(),
            nb::arg("output") = std::nullopt,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("compute_kernel_config") = std::nullopt,
        });
}

void bind_moreh_cumsum_backward_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::moreh_cumsum_backward,
        "Moreh Cumsum Backward Operation",
        ttnn::nanobind_arguments_t{
            nb::arg("output_grad"),
            nb::arg("dim"),
            nb::kw_only(),
            nb::arg("input_grad") = std::nullopt,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("compute_kernel_config") = std::nullopt,
        });
}

}  // namespace ttnn::operations::moreh::moreh_cumsum
