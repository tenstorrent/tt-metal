// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::experimental::slice_write {
namespace py = pybind11;

void bind_slice_write(py::module& module);

}  // namespace ttnn::operations::experimental::slice_write
