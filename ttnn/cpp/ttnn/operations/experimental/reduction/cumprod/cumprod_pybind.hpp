// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pybind11/pybind11.h"

namespace ttnn::operations::experimental::reduction::cumprod::detail {
namespace py = pybind11;
void bind_cumprod_operation(py::module& module);

}  // namespace ttnn::operations::experimental::reduction::cumprod::detail
