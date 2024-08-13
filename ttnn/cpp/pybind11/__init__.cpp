// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core.hpp"
#include "device.hpp"
#include "multi_device.hpp"
#include "types.hpp"
#include "reports.hpp"
#include "activation.hpp"
#include "export_enum.hpp"
#include "json_class.hpp"

#include "ttnn/deprecated/tt_lib/csrc/tt_lib_bindings.hpp"
#include "operations/__init__.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_ttnn, module) {
    module.doc() = "Python bindings for TTNN";

    auto m_deprecated = module.def_submodule("deprecated", "Contains deprecated tt_lib bindings for tensor, device, profiler");
    tt::bind_deprecated(m_deprecated);

    auto m_types = module.def_submodule("types", "ttnn Types");
    ttnn::types::py_module(m_types);

    auto m_activation = module.def_submodule("activation", "ttnn Activation");
    ttnn::activation::py_module(m_activation);

    auto m_core = module.def_submodule("core", "core functions");
    ttnn::core::py_module(m_core);

    auto m_device = module.def_submodule("device", "ttnn devices");
    ttnn::device::py_module(m_device);

    auto m_multi_device = module.def_submodule("multi_device", "ttnn multi_device");
    ttnn::multi_device::py_module(m_multi_device);

    auto m_reports = module.def_submodule("reports", "ttnn reports");
    ttnn::reports::py_module(m_reports);

    auto m_operations = module.def_submodule("operations", "ttnn Operations");
    ttnn::operations::py_module(m_operations);

    module.attr("CONFIG") = &ttnn::CONFIG;

    module.def("get_operation_id", []() { return ttnn::OPERATION_ID; }, "Get operation id");
    module.def("set_operation_id", [](int operation_id) { ttnn::OPERATION_ID = operation_id; }, "Set operation id");
    module.def("increment_operation_id", []() { ttnn::OPERATION_ID++; }, "Increment operation id");
}
