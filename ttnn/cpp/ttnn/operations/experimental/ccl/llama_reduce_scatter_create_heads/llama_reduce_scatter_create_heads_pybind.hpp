// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::experimental::ccl {

void py_bind_llama_rs_create_heads(pybind11::module& module);

}  // namespace ttnn::operations::experimental::ccl
