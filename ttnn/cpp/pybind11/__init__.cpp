// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/__init__.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "activation.hpp"
#include "core.hpp"
#include "device.hpp"
#include "profiler.hpp"
#include "events.hpp"
#include "tensor.hpp"
#include "reports.hpp"
#include "ttnn/distributed/distributed_pybind.hpp"
#include "ttnn/deprecated/tt_lib/csrc/operations/primary/module.hpp"
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
    auto m_deprecated = module.def_submodule("deprecated", "Deprecated tt_lib bindings");
    auto m_tensor = module.def_submodule("tensor", "ttnn tensor");

    auto m_depr_operations = m_deprecated.def_submodule("operations", "Submodule for experimental operations");
    auto m_primary_ops = m_depr_operations.def_submodule("primary", "Primary operations");

    auto m_graph = module.def_submodule("graph", "Contains graph capture functions");
    auto m_types = module.def_submodule("types", "ttnn Types");
    auto m_activation = module.def_submodule("activation", "ttnn Activation");
    auto m_core = module.def_submodule("core", "core functions");
    auto m_device = module.def_submodule("device", "ttnn devices");
    auto m_multi_device = module.def_submodule("multi_device", "ttnn multi_device");
    auto m_events = module.def_submodule("events", "ttnn events");
    auto m_profiler = module.def_submodule("profiler", "Submodule defining the profiler");
    auto m_reports = module.def_submodule("reports", "ttnn reports");
    auto m_operations = module.def_submodule("operations", "ttnn Operations");

    // TYPES
    ttnn::tensor::tensor_mem_config_module_types(m_tensor);
    ttnn::tensor::pytensor_module_types(m_tensor);
    ttnn::graph::py_graph_module_types(m_graph);


    ttnn::types::py_module_types(m_types);
    ttnn::activation::py_module_types(m_activation);
    ttnn::core::py_module_types(m_core);
    ttnn::device::py_device_module_types(m_device);
    ttnn::distributed::py_module_types(m_multi_device);
    ttnn::events::py_module_types(m_events);
    ttnn::reports::py_module_types(m_reports);

    // FUNCTIONS / OPERATIONS
    ttnn::tensor::tensor_mem_config_module(m_tensor);
    ttnn::tensor::pytensor_module(m_tensor);
    ttnn::core::py_module(m_core);
    ttnn::graph::py_graph_module(m_graph);


#if defined(TRACY_ENABLE)
    py::function tracy_decorator = py::module::import("tracy.ttnn_profiler_wrapper").attr("callable_decorator");

    tracy_decorator(m_device);
    tracy_decorator(m_tensor);
    tracy_decorator(m_depr_operations);
#endif

    ttnn::types::py_module(m_types);
    ttnn::activation::py_module(m_activation);
    ttnn::device::py_device_module(m_device);
    ttnn::distributed::py_module(m_multi_device);
    ttnn::events::py_module(m_events);
    ttnn::profiler::py_module(m_profiler);
    ttnn::reports::py_module(m_reports);

    // ttnn operations have to come before the deprecated ones,
    // because ttnn defines additional type bindings.
    // TODO: pull them out of the ttnn::operations::py_module.
    ttnn::operations::py_module(m_operations);
    tt::operations::primary::py_module(m_primary_ops);

    module.attr("CONFIG") = &ttnn::CONFIG;
    module.def("get_python_operation_id", []()->std::uint64_t {return ttnn::CoreIDs::instance().get_python_operation_id();} , "Get operation id");
    module.def("set_python_operation_id", [](std::uint64_t id){ttnn::CoreIDs::instance().set_python_operation_id(id);}, "Set operation id");
    module.def("fetch_and_increment_python_operation_id", []()->std::uint64_t {return ttnn::CoreIDs::instance().fetch_and_increment_python_operation_id();}, "Increment tensor id and return the previously held id");

    module.def("get_tensor_id", []()->std::uint64_t {return ttnn::CoreIDs::instance().get_tensor_id();}, "Get tensor id");
    module.def("set_tensor_id", [](std::uint64_t id){ttnn::CoreIDs::instance().set_tensor_id(id);}, "Set tensor id");
    module.def("fetch_and_increment_tensor_id", []()->std::uint64_t {return ttnn::CoreIDs::instance().fetch_and_increment_tensor_id();}, "Increment tensor id and return the previously held id");

    module.def("get_device_operation_id", []()->std::uint64_t {return ttnn::CoreIDs::instance().get_device_operation_id();}, "Get device operation id");
    module.def("set_device_operation_id", [](std::uint64_t id){ttnn::CoreIDs::instance().set_device_operation_id(id);}, "Set device operation id");
    module.def("fetch_and_increment_device_operation_id", []()->std::uint64_t {return ttnn::CoreIDs::instance().fetch_and_increment_device_operation_id();}, "Increment device operation id and return the previously held id");
}
