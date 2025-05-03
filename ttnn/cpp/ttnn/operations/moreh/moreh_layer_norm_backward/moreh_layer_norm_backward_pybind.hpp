// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pybind11/pybind_fwd.hpp"

namespace py = pybind11;

namespace ttnn::operations::moreh::moreh_layer_norm_backward {
void bind_moreh_layer_norm_backward_operation(py::module& module);
}  // namespace ttnn::operations::moreh::moreh_layer_norm_backward
