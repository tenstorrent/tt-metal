// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_sum_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/moreh/moreh_sum/moreh_sum.hpp"

namespace ttnn::operations::moreh::moreh_sum {
void bind_moreh_sum_operation(nb::module_& mod) {
    const auto* doc = "Moreh Sum Operation";

    ttnn::bind_function<"moreh_sum">(
        mod,
        doc,
        ttnn::overload_t(
            &ttnn::moreh_sum,
            nb::arg("input"),
            nb::arg("dim") = nb::none(),
            nb::kw_only(),
            nb::arg("keepdim") = false,
            nb::arg("output") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()));
}
}  // namespace ttnn::operations::moreh::moreh_sum
