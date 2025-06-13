// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::binary_backward {

namespace py = pybind11;
void py_module(py::module& module);

}  // namespace ttnn::operations::binary_backward
