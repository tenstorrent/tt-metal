// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::tensor {

namespace py = pybind11;

void tensor_mem_config_module_types(py::module& module);
void tensor_mem_config_module(py::module& module);

}  // namespace ttnn::tensor
