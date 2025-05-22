// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace py = pybind11;

namespace ttnn::operations::moreh::moreh_cumsum {
void bind_moreh_cumsum_operation(py::module& module);
void bind_moreh_cumsum_backward_operation(py::module& module);
}  // namespace ttnn::operations::moreh::moreh_cumsum
