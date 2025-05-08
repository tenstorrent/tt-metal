// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"

#include "pad.hpp"

namespace ttnn::operations::data_movement::detail {
namespace py = pybind11;

void bind_pad(py::module& module);
}  // namespace ttnn::operations::data_movement::detail
