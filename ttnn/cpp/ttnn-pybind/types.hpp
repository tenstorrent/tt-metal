// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>

#include "small_vector_caster.hpp"  // NOLINT - for pybind11 SmallVector binding support.

namespace py = pybind11;

namespace ttnn {
namespace types {

void py_module_types(py::module& module);
void py_module(py::module& module);

}  // namespace types
}  // namespace ttnn
