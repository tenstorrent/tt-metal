// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pybind11/pybind_fwd.hpp"

namespace py = pybind11;

namespace ttnn::operations::experimental::deinterleave {
void bind_deinterleave_operation(py::module& module);
}  // namespace ttnn::operations::experimental::deinterleave
