// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pybind11/pybind_fwd.hpp"
#include <string>

namespace py = pybind11;

namespace ttnn::operations::binary_ng {
namespace detail {
template <typename T>
void bind_binary_ng_operation(py::module& module, T op, const std::string& docstring);
}

void py_module(py::module& module);
}  // namespace ttnn::operations::binary_ng
