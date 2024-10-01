// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "uniform_pybind.hpp"

#include <pybind11/pybind11.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "uniform.hpp"

namespace py = pybind11;

namespace ttnn::operations::uniform {

void bind_uniform_operation(py::module &module) {
    bind_registered_operation(
        module,
        ttnn::uniform,
        "Uniform",
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::arg("from") = 0,
            py::arg("to") = 1,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});
}
}  // namespace ttnn::operations::uniform
