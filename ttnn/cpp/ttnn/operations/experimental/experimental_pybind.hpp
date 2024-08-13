// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pybind11/pybind_fwd.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace ttnn::operations::experimental {
void py_module(pybind11::module& module);

}  // namespace ttnn::operations::experimental
