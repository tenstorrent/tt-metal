// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "unary.hpp"
#include "binary.hpp"
#include "core.hpp"
#include "matmul.hpp"
#include "transformer.hpp"
#include "normalization.hpp"

namespace py = pybind11;

namespace ttnn {

namespace operations {

void py_module(py::module& module) {
    auto m_unary = module.def_submodule("unary", "unary operations");
    unary::py_module(m_unary);

    auto m_binary = module.def_submodule("binary", "binary operations");
    binary::py_module(m_binary);

    auto m_core = module.def_submodule("core", "core operations");
    core::py_module(m_core);

    auto m_matmul = module.def_submodule("matmul", "matmul operations");
    matmul::py_module(m_matmul);

    auto m_transformer = module.def_submodule("transformer", "transformer operations");
    transformer::py_module(m_transformer);

    auto m_normalization = module.def_submodule("normalization", "normalization operations");
    normalization::py_module(m_normalization);
}

}  // namespace operations

}  // namespace ttnn
