// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::loss {
namespace py = pybind11;
void py_bind_loss_functions(py::module& module);

}  // namespace ttnn::operations::loss
