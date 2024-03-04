// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// #pragma once
#include "operations/__init__.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core.hpp"
#include "device.hpp"
#include "types.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_ttnn, m_ttnn) {
    // m_ttnn.attr("__name__") = "_ttnn";
    m_ttnn.doc() = "Python bindings for TTNN";

    auto m_types = m_ttnn.def_submodule("types", "ttnn Types");
    ttnn::types::py_module(m_types);

    auto m_core = m_ttnn.def_submodule("core", "core functions");
    ttnn::core::py_module(m_core);

    auto m_device = m_ttnn.def_submodule("device", "ttnn devices");
    ttnn::device::py_module(m_device);

    auto m_operations = m_ttnn.def_submodule("operations", "ttnn Operations");
    ttnn::operations::py_module(m_operations);
}
