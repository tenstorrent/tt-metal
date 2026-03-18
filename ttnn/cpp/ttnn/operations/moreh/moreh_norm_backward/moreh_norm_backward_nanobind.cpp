// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_norm_backward_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "moreh_norm_backward.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::moreh::moreh_norm_backward {
void bind_moreh_norm_backward_operation(nb::module_& mod) {
    const auto* doc = "Moreh Norm Backward Operation";

    ttnn::bind_function<"moreh_norm_backward">(
        mod,
        doc,
        ttnn::overload_t(
            &ttnn::moreh_norm_backward,
            nb::arg("input"),
            nb::arg("output"),
            nb::arg("output_grad"),
            nb::arg("p"),
            nb::kw_only(),
            nb::arg("dim") = nb::none(),
            nb::arg("keepdim") = false,
            nb::arg("input_grad") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()));
}
}  // namespace ttnn::operations::moreh::moreh_norm_backward
