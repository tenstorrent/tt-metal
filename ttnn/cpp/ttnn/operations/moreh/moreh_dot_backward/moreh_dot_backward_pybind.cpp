// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_dot_backward_pybind.hpp"

#include <optional>

#include "moreh_dot_backward.hpp"
#include "pybind11/cast.h"
#include "pybind11/decorators.hpp"
#include "ttnn/operations/moreh/moreh_dot_backward/device/moreh_dot_backward_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_dot_backward {
void bind_moreh_dot_backward_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::moreh_dot_backward,
        "Moreh Dot Backward Operation",
        ttnn::pybind_arguments_t{
            py::arg("output_grad"),
            py::arg("input"),
            py::arg("other"),
            py::kw_only(),
            py::arg("input_grad") = std::nullopt,
            py::arg("other_grad") = std::nullopt,
            py::arg("memory_config") = std::nullopt});
}
}  // namespace ttnn::operations::moreh::moreh_dot_backward
