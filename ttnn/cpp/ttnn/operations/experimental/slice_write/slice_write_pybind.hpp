// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pybind11/pybind_fwd.hpp"

namespace ttnn::operations::experimental::slice_write {
namespace py = pybind11;

void bind_slice_write(py::module& module);

}  // namespace ttnn::operations::experimental::slice_write
