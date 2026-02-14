// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_sum_backward_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "moreh_sum_backward.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::moreh::moreh_sum_backward {
void bind_moreh_sum_backward_operation(nb::module_& mod) {
    const auto* doc = R"doc(
        Moreh Sum Backward Operation
    )doc";

    ttnn::bind_function<"moreh_sum_backward">(
        mod,
        doc,
        ttnn::overload_t(
            &ttnn::moreh_sum_backward,
            nb::arg("output_grad"),
            nb::kw_only(),
            nb::arg("input") = nb::none(),
            nb::arg("dim") = nb::none(),
            nb::arg("keepdim") = false,
            nb::arg("input_grad") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()));
}
}  // namespace ttnn::operations::moreh::moreh_sum_backward
