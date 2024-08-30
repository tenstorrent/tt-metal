// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>

#include "ttnn/operations/moreh/moreh_adam/moreh_adam_pybind.hpp"

namespace ttnn::operations::moreh {
void py_module(py::module& module) { moreh_adam::bind_moreh_adam_operation(module); }
}  // namespace ttnn::operations::moreh
