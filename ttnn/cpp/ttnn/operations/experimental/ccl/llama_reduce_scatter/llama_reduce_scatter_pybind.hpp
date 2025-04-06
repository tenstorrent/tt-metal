// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/pybind11/decorators.hpp"

namespace ttnn::operations::experimental::ccl {

void py_bind_llama_reduce_scatter(pybind11::module& module);

}  // namespace ttnn::operations::experimental::ccl
