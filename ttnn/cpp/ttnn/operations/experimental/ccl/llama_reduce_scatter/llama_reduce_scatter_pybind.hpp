// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/pybind11/decorators.hpp"

namespace ttnn::operations::experimental::ccl {
namespace py = pybind11;

void py_bind_llama_reduce_scatter(py::module& module);

}  // namespace ttnn::operations::experimental::ccl
