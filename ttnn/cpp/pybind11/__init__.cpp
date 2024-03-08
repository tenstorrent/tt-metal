// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// #pragma once
#include "operations/__init__.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core.hpp"
#include "device.hpp"
#include "multi_device.hpp"
#include "types.hpp"
#include "reports.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_ttnn, m_ttnn) {
    m_ttnn.doc() = "Python bindings for TTNN";

    auto m_types = m_ttnn.def_submodule("types", "ttnn Types");
    ttnn::types::py_module(m_types);

    auto m_core = m_ttnn.def_submodule("core", "core functions");
    ttnn::core::py_module(m_core);

    auto m_device = m_ttnn.def_submodule("device", "ttnn devices");
    ttnn::device::py_module(m_device);

    auto m_multi_device = m_ttnn.def_submodule("multi_device", "ttnn multi_device");
    ttnn::multi_device::py_module(m_multi_device);
    
    auto m_reports = m_ttnn.def_submodule("reports", "ttnn reports");
    ttnn::reports::py_module(m_reports);

    auto m_operations = m_ttnn.def_submodule("operations", "ttnn Operations");
    ttnn::operations::py_module(m_operations);
}
