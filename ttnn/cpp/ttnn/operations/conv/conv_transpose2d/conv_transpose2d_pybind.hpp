// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::conv::conv_transpose2d {
namespace py = pybind11;
void py_bind_conv_transpose2d(py::module& module);
}
