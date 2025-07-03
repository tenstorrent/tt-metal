// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::experimental::reduction::detail {
namespace py = pybind11;
void bind_argmax_operation(py::module& module);
void bind_argmin_operation(py::module& module);

}  // namespace ttnn::operations::experimental::reduction::detail
