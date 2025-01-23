// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pybind11/pybind_fwd.hpp"

namespace py = pybind11;

namespace ttnn::operations::experimental::broadcast_to::detail {
void py_bind_module(py::module& module);
}  // namespace ttnn::operations::experimental::broadcast_to::detail
