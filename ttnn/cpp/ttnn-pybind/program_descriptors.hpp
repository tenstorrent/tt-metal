// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::program_descriptors {

namespace py = pybind11;
void py_module_types(py::module& module);

}  // namespace ttnn::program_descriptors
