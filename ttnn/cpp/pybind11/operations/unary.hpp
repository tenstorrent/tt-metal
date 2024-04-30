// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/operations/unary.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace unary {

void py_module(py::module& module) {
    module.def("silu", &silu, py::arg("input_tensor"), py::kw_only(), py::arg("memory_config") = std::nullopt);
}

}  // namespace unary
}  // namespace operations
}  // namespace ttnn
