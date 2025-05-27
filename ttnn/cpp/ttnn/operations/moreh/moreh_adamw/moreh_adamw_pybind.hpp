// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::moreh::moreh_adamw {
namespace py = pybind11;

void bind_moreh_adamw_operation(py::module& module);

void py_module(py::module& module);

}  // namespace ttnn::operations::moreh::moreh_adamw
