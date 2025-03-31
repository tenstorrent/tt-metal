// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybind11/decorators.hpp"

namespace py = pybind11;

namespace ttnn::operations::experimental::reduction {

void bind_cumsum_operation(py::module& module);
}
