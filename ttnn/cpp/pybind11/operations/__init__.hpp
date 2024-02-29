// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "binary.hpp"
#include "core.hpp"
#include "matmul.hpp"

namespace py = pybind11;

namespace ttnn {

namespace operations {

void py_module(py::module& m_operations) {
    auto m_binary = m_operations.def_submodule("binary", "binary operations");
    binary::py_module(m_binary);

    auto m_core = m_operations.def_submodule("core", "core operations");
    core::py_module(m_core);

    auto m_matmul = m_operations.def_submodule("matmul", "matmul operations");
    matmul::py_module(m_matmul);
}

}  // namespace operations

}  // namespace ttnn
