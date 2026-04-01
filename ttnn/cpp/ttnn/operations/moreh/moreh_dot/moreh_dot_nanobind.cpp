// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_dot_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "moreh_dot.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::moreh::moreh_dot {

void bind_moreh_dot_operation(nb::module_& mod) {
    const auto* doc = "Moreh Moreh Dot Operation";

    ttnn::bind_function<"moreh_dot">(
        mod,
        doc,
        ttnn::overload_t(
            &ttnn::moreh_dot,
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("output") = nb::none(),
            nb::arg("dtype") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()));
}

}  // namespace ttnn::operations::moreh::moreh_dot
