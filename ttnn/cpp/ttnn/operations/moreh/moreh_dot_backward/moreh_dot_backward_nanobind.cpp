// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_dot_backward_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "moreh_dot_backward.hpp"

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/moreh/moreh_dot_backward/device/moreh_dot_backward_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_dot_backward {
void bind_moreh_dot_backward_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::moreh_dot_backward,
        "Moreh Dot Backward Operation",
        ttnn::nanobind_arguments_t{
            nb::arg("output_grad"),
            nb::arg("input"),
            nb::arg("other"),
            nb::kw_only(),
            nb::arg("input_grad") = nb::none(),
            nb::arg("other_grad") = nb::none(),
            nb::arg("memory_config") = nb::none()});
}
}  // namespace ttnn::operations::moreh::moreh_dot_backward
