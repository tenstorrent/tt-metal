// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::tensor {

namespace py = pybind11;
void pytensor_module_types(py::module& m_tensor);

void pytensor_module(py::module& m_tensor);

}  // namespace ttnn::tensor
