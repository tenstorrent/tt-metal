// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "full_like_op_bind.hpp"

#include "full_like_pybind.hpp"

namespace ttnn::operations::full_like {
    void py_module(py::module& module) {
        bind_full_like_operation(module);
    }
}
