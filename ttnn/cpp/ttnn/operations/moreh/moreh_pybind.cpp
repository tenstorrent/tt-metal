// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/moreh/moreh_dot_op/moreh_dot_pybind.hpp"

namespace ttnn::operations::moreh {
void py_module(py::module& module) {
    moreh_dot::bind_moreh_dot_operation(module);
}
}  // namespace ttnn::operations::moreh
