// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace py = pybind11;

namespace ttnn::operations::experimental::adaptive_pool {

void bind_adaptive_avg_pool2d_operation(py::module& module);
void bind_adaptive_max_pool2d_operation(py::module& module);

}  // namespace ttnn::operations::experimental::adaptive_pool
