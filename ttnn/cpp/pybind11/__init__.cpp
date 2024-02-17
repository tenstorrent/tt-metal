// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// #pragma once

#include "operations/__init__.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "types.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_ttnn, m_ttnn) {
    // m_ttnn.attr("__name__") = "_ttnn";
    m_ttnn.doc() = "Python bindings for TTNN";

    auto m_types = m_ttnn.def_submodule("types", "ttnn Types");
    ttnn::types::py_module(m_types);

    auto m_operations = m_ttnn.def_submodule("operations", "ttnn Operations");
    ttnn::operations::py_module(m_operations);

#ifdef TTNN_ENABLE_LOGGING
    m_ttnn.attr("TTNN_ENABLE_LOGGING") = true;
#else
    m_ttnn.attr("TTNN_ENABLE_LOGGING") = false;
#endif
}
