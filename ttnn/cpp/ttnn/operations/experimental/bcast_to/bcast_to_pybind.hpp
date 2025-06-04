// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::experimental::broadcast_to::detail {
namespace py = pybind11;
void py_bind_broadcast_to(py::module& module);
}  // namespace ttnn::operations::experimental::broadcast_to::detail
