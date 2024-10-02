// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pybind11/pybind_fwd.hpp"

namespace py = pybind11;

namespace ttnn::operations::moreh::moreh_sgd {
void bind_moreh_sgd_operation(py::module& module);
}  // namespace ttnn::operations::moreh::moreh_sgd
