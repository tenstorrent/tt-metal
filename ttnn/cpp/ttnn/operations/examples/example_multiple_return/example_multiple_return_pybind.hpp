// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::examples {

namespace py = pybind11;
void bind_example_multiple_return_operation(py::module& module);

}  // namespace ttnn::operations::examples
