// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_dot_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "device/moreh_dot_device_operation.hpp"
#include "moreh_dot.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"

namespace py = pybind11;

namespace ttnn::operations::moreh::moreh_dot {

void bind_moreh_dot_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::moreh_dot,
        "Moreh Moreh Dot Operation",
        ttnn::pybind_arguments_t{
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("output") = std::nullopt,
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});
}

}  // namespace ttnn::operations::moreh::moreh_dot
