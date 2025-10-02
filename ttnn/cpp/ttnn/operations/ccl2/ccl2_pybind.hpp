// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/operations/ccl2/all_gather/all_gather_pybind.hpp"

namespace py = pybind11;

namespace ttnn::operations::ccl2 {

void py_module(py::module& module);

}  // namespace ttnn::operations::ccl2
