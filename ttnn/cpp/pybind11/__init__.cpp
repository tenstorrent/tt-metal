// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core.hpp"
#include "device.hpp"
#include "events.hpp"
#include "multi_device.hpp"
#include "types.hpp"
#include "reports.hpp"
#include "activation.hpp"
#include "export_enum.hpp"
#include "json_class.hpp"

#include "ttnn/deprecated/tt_lib/csrc/tt_lib_bindings.hpp"
#include "operations/__init__.hpp"

#include "ttnn/graph/graph_pybind.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_ttnn, module) {
    module.doc() = "Python bindings for TTNN";

    auto m_graph = module.def_submodule("graph", "Contains graph capture functions");
    ttnn::graph::py_graph_module(m_graph);

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

    auto m_events = module.def_submodule("events", "ttnn events");
    ttnn::events::py_module(m_events);

    auto m_reports = module.def_submodule("reports", "ttnn reports");
    ttnn::reports::py_module(m_reports);

    auto m_operations = module.def_submodule("operations", "ttnn Operations");
    ttnn::operations::py_module(m_operations);

    module.attr("CONFIG") = &ttnn::CONFIG;

    module.def("get_python_operation_id", ttnn::get_python_operation_id, "Get operation id");
    module.def("set_python_operation_id", ttnn::set_python_operation_id, "Set operation id");
    module.def("increment_python_operation_id", ttnn::increment_python_operation_id, "Increment operation id");

    module.def("get_tensor_id", ttnn::get_tensor_id, "Get tensor id");
    module.def("set_tensor_id", ttnn::set_tensor_id, "Set tensor id");
    module.def("increment_tensor_id", ttnn::increment_tensor_id, "Increment tensor id");

    module.def("get_device_operation_id", ttnn::get_device_operation_id, "Get device operation id");
    module.def("set_device_operation_id", ttnn::set_device_operation_id, "Set device operation id");
    module.def("increment_device_operation_id", ttnn::increment_device_operation_id, "Increment device operation id");
}
