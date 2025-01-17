// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/pybind11/decorators.hpp"

#include "transpose.hpp"

namespace ttnn::operations::data_movement::detail {
namespace py = pybind11;

void bind_transpose(py::module& module);
}  // namespace ttnn::operations::data_movement::detail
