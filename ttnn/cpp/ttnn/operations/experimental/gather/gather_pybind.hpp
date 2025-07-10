// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::experimental::gather::detail {
namespace py = pybind11;

void bind_gather_operation(py::module& module);

}  // namespace ttnn::operations::experimental::gather::detail
