// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_dot_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "moreh_dot.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace ttnn::operations::moreh::moreh_dot {

void bind_moreh_dot_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::moreh_dot,
        "Moreh Moreh Dot Operation",
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("output") = std::nullopt,
            nb::arg("dtype") = std::nullopt,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("compute_kernel_config") = std::nullopt});
}

}  // namespace ttnn::operations::moreh::moreh_dot
