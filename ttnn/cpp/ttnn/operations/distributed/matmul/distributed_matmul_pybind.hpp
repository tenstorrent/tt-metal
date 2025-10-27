// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ttnn::operations::distributed {

void py_bind_distributed_matmul(py::module& module);

}  // namespace ttnn::operations::distributed
