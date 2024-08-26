// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/__init__.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "activation.hpp"
#include "core.hpp"
#include "device.hpp"
#include "events.hpp"
#include "multi_device.hpp"
#include "reports.hpp"
#include "ttnn/deprecated/tt_lib/csrc/operations/primary/module.hpp"
#include "ttnn/deprecated/tt_lib/csrc/tt_lib_bindings.hpp"
#include "ttnn/deprecated/tt_lib/csrc/tt_lib_bindings_tensor.hpp"
#include "ttnn/graph/graph_pybind.hpp"
#include "types.hpp"

PYBIND11_MODULE(_ttnn, module) {
    module.doc() = "Python bindings for TTNN";

    /*
    We have to make sure every class and enum is bound before any function that uses it as an argument or a return type.
    So we split the binding calls into two parts: one for classes and enums, and one for functions.
    Another issue to be aware of is that we have to define each shared submodule only once. Therefore, all def_submodule calls
    have to be put in here.
    */

    // MODULES
    auto m_deprecated = module.def_submodule("deprecated", "Deprecated tt_lib bindings for tensor, device, profiler");
    auto m_depr_tensor = m_deprecated.def_submodule("tensor", "Submodule defining an tt_metal tensor");
    auto m_depr_device = m_deprecated.def_submodule("device", "Submodule defining a host or device");

    auto m_depr_operations = m_deprecated.def_submodule("operations", "Submodule for experimental operations");
    auto m_primary_ops = m_depr_operations.def_submodule("primary", "Primary operations");

    auto m_profiler = m_deprecated.def_submodule("profiler", "Submodule defining the profiler");

    auto m_graph = module.def_submodule("graph", "Contains graph capture functions");
    auto m_types = module.def_submodule("types", "ttnn Types");
    auto m_activation = module.def_submodule("activation", "ttnn Activation");
    auto m_core = module.def_submodule("core", "core functions");
    auto m_device = module.def_submodule("device", "ttnn devices");
    auto m_multi_device = module.def_submodule("multi_device", "ttnn multi_device");
    auto m_events = module.def_submodule("events", "ttnn events");
    auto m_reports = module.def_submodule("reports", "ttnn reports");
    auto m_operations = module.def_submodule("operations", "ttnn Operations");

    // TYPES
    ttnn::graph::py_graph_module_types(m_graph);

    tt::tt_metal::TensorModuleTypes(m_depr_tensor);
    tt::tt_metal::DeviceModuleTypes(m_depr_device);
    tt::operations::primary::py_module_types(m_primary_ops);

    ttnn::types::py_module_types(m_types);
    ttnn::activation::py_module_types(m_activation);
    ttnn::core::py_module_types(m_core);
    ttnn::multi_device::py_module_types(m_multi_device);
    ttnn::events::py_module_types(m_events);
    ttnn::reports::py_module_types(m_reports);

    // FUNCTIONS / OPERATIONS
    ttnn::core::py_module(m_core);
    ttnn::graph::py_graph_module(m_graph);

    tt::tt_metal::TensorModule(m_depr_tensor);
    tt::tt_metal::DeviceModule(m_depr_device);
    tt::tt_metal::ProfilerModule(m_profiler);

#if defined(TRACY_ENABLE)
    py::function tracy_decorator = py::module::import("tracy.ttnn_profiler_wrapper").attr("callable_decorator");

    tracy_decorator(m_depr_device);
    tracy_decorator(m_depr_tensor);
    tracy_decorator(m_depr_operations);
#endif

    ttnn::types::py_module(m_types);
    ttnn::activation::py_module(m_activation);
    ttnn::device::py_module(m_device);
    ttnn::multi_device::py_module(m_multi_device);
    ttnn::events::py_module(m_events);
    ttnn::reports::py_module(m_reports);

    // ttnn operations have to come before the deprecated ones,
    // because ttnn defines additional type bindings.
    // TODO: pull them out of the ttnn::operations::py_module.
    ttnn::operations::py_module(m_operations);
    tt::operations::primary::py_module(m_primary_ops);

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
