// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_cumsum_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/moreh/moreh_cumsum/moreh_cumsum.hpp"

namespace ttnn::operations::moreh::moreh_cumsum {

void bind_moreh_cumsum_operation(nb::module_& mod) {
    ttnn::bind_function<"moreh_cumsum">(
        mod,
        "Moreh Cumsum Operation",
        ttnn::overload_t(
            &ttnn::moreh_cumsum,
            nb::arg("input"),
            nb::arg("dim"),
            nb::kw_only(),
            nb::arg("output") = nb::none(),
            nb::arg("memory_config") = nb::none()));
}

void bind_moreh_cumsum_backward_operation(nb::module_& mod) {
    ttnn::bind_function<"moreh_cumsum_backward">(
        mod,
        "Moreh Cumsum Backward Operation",
        ttnn::overload_t(
            &ttnn::moreh_cumsum_backward,
            nb::arg("output_grad"),
            nb::arg("dim"),
            nb::kw_only(),
            nb::arg("input_grad") = nb::none(),
            nb::arg("memory_config") = nb::none()));
}

}  // namespace ttnn::operations::moreh::moreh_cumsum
