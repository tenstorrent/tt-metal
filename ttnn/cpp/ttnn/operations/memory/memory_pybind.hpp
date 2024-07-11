// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/memory/memory.hpp"

#include "ttnn/types.hpp"

namespace py = pybind11;
namespace ttnn {
namespace operations {
namespace memory {


void py_module(py::module& module) {
       module.def(
        "read_tensor_from_L1",
        &read_tensor_from_L1,
        py::arg("addr").noconvert(),
        py::arg("core").noconvert(),
        py::arg("size").noconvert(),
        py::arg("dtype").noconvert(),
        py::arg("device").noconvert()
       );

       module.def(
       "print_tensor_info",
       &print_tensor_info,
       py::arg("tensor").noconvert());
}

}
}
}
