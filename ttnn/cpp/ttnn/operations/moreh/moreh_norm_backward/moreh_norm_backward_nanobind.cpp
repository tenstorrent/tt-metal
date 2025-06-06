// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_norm_backward_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "moreh_norm_backward.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace nb = nanobind;

namespace ttnn::operations::moreh::moreh_norm_backward {
void bind_moreh_norm_backward_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::moreh_norm_backward,
        "Moreh Norm Backward Operation",
        ttnn::nanobind_arguments_t{
            nb::arg("input"),
            nb::arg("output"),
            nb::arg("output_grad"),
            nb::arg("p"),
            nb::kw_only(),
            nb::arg("dim") = std::nullopt,
            nb::arg("keepdim") = false,
            nb::arg("input_grad") = std::nullopt,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("compute_kernel_config") = std::nullopt,
        });
}
}  // namespace ttnn::operations::moreh::moreh_norm_backward
