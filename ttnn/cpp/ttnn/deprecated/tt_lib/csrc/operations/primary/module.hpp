// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace tt {
namespace operations {
namespace primary {

void py_module(py::module& m_primary) {}

}  // namespace
   // primary
}  // namespace
   // operations
}  // namespace
   // tt
