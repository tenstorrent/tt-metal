// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-nanobind/operations/__init__.hpp"

#include <cstdint>

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/activation.hpp"
#include "ttnn-nanobind/cluster.hpp"
#include "ttnn-nanobind/core.hpp"
#include "ttnn-nanobind/device.hpp"
#include "ttnn-nanobind/events.hpp"
#include "ttnn-nanobind/fabric.hpp"
#include "ttnn-nanobind/global_circular_buffer.hpp"
#include "ttnn-nanobind/global_semaphore.hpp"
#include "ttnn-nanobind/hd_socket.hpp"
#include "ttnn-nanobind/h2d_stream_service.hpp"
#include "ttnn-nanobind/mesh_socket.hpp"
#include "ttnn-nanobind/bfp_utils.hpp"
#include "ttnn-nanobind/operations/copy.hpp"
#include "ttnn-nanobind/operations/core.hpp"
#include "ttnn-nanobind/operations/trace.hpp"
#include "ttnn-nanobind/profiler.hpp"
#include "ttnn-nanobind/program_descriptors.hpp"
#include "ttnn-nanobind/tensor_accessor_args.hpp"
#include "ttnn-nanobind/reports.hpp"
#include "ttnn-nanobind/tensor.hpp"
#include "ttnn-nanobind/types.hpp"

#include "ttnn/core.hpp"
#include "ttnn/distributed/distributed_nanobind.hpp"
#include "ttnn/graph/graph_nanobind.hpp"
// --- NUKED OPS: only keep-set category nanobind headers remain ---
#include "ttnn/operations/creation/creation_nanobind.hpp"
#include "ttnn/operations/data_movement/data_movement_nanobind.hpp"
#include "ttnn/operations/examples/examples_nanobind.hpp"
#include "ttnn/operations/generic/generic_op_nanobind.hpp"

namespace nb = nanobind;

namespace ttnn::operations {

void py_module(nb::module_& mod) {
    nb::set_leak_warnings(true);

    auto m_core = mod.def_submodule("core", "core operations");
    core::py_module_types(m_core);
    core::py_module(m_core);

    auto m_trace = mod.def_submodule("trace", "trace operations");
    trace::py_module_types(m_trace);
    trace::py_module(m_trace);

    auto m_examples = mod.def_submodule("examples", "examples of operations");
    examples::py_module(m_examples);

    // --- NUKED OPS: only keep-set categories registered below ---
    auto m_creation = mod.def_submodule("creation", "creation operations");
    creation::bind_creation_operations(m_creation);

    auto m_data_movement = mod.def_submodule("data_movement", "data_movement operations");
    data_movement::py_module(m_data_movement);

    auto m_copy = mod.def_submodule("copy", "copy operations");
    copy::py_module(m_copy);

    auto m_generic = mod.def_submodule("generic", "ttnn generic operation interface");
    generic::bind_generic_operation(m_generic);
}
}  // namespace ttnn::operations

NB_MODULE(_ttnn, mod) {
    mod.doc() = "Python bindings for TTNN";

    /*
    We have to make sure every class and enum is bound before any function that uses it as an argument or a return type.
    So we split the binding calls into two parts: one for classes and enums, and one for functions.
    Another issue to be aware of is that we have to define each shared submodule only once. Therefore, all def_submodule
    calls have to be put in here.
    */

    // MODULES
    auto m_deprecated = mod.def_submodule("deprecated", "Deprecated tt_lib bindings");
    auto m_tensor = mod.def_submodule("tensor", "ttnn tensor");

    auto m_depr_operations = m_deprecated.def_submodule("operations", "Submodule for experimental operations");
    auto m_primary_ops = m_depr_operations.def_submodule("primary", "Primary operations");

    auto m_graph = mod.def_submodule("graph", "Contains graph capture functions");
    auto m_types = mod.def_submodule("types", "ttnn Types");
    auto m_activation = mod.def_submodule("activation", "ttnn Activation");
    auto m_cluster = mod.def_submodule("cluster", "ttnn cluster");
    auto m_core = mod.def_submodule("core", "core functions");
    auto m_device = mod.def_submodule("device", "ttnn devices");
    auto m_multi_device = mod.def_submodule("multi_device", "ttnn multi_device");
    auto m_events = mod.def_submodule("events", "ttnn events");
    auto m_global_circular_buffer = mod.def_submodule("global_circular_buffer", "ttnn global circular buffer");
    auto m_global_semaphore = mod.def_submodule("global_semaphore", "ttnn global semaphore");
    auto m_hd_socket = mod.def_submodule("hd_socket", "ttnn host-device sockets");
    auto m_h2d_stream_service =
        mod.def_submodule("h2d_stream_service", "ttnn persistent host-to-device streaming service");
    auto m_mesh_socket = mod.def_submodule("mesh_socket", "ttnn mesh socket");
    auto m_profiler = mod.def_submodule("profiler", "Submodule defining the profiler");
    auto m_reports = mod.def_submodule("reports", "ttnn reports");
    auto m_operations = mod.def_submodule("operations", "ttnn Operations");
    auto m_fabric = mod.def_submodule("fabric", "Fabric instantiation APIs");
    auto m_program_descriptors = mod.def_submodule("program_descriptor", "Program descriptors types");
    auto m_tensor_accessor_args = mod.def_submodule("tensor_accessor_args", "Tensor accessor args types");

    // TYPES
    ttnn::tensor::tensor_mem_config_module_types(m_tensor);
    ttnn::tensor::pytensor_module_types(m_tensor);
    ttnn::graph::py_graph_module_types(m_graph);

    ttnn::types::py_module_types(m_types);
    ttnn::activation::py_module_types(m_activation);
    ttnn::cluster::py_cluster_module_types(m_cluster);
    ttnn::core::py_module_types(m_core);
    ttnn::device::py_device_module_types(m_device);
    ttnn::fabric::bind_fabric_api(m_fabric);
    ttnn::distributed::py_module_types(m_multi_device);
    ttnn::events::py_module_types(m_events);
    ttnn::global_circular_buffer::py_module_types(m_global_circular_buffer);
    ttnn::global_semaphore::py_module_types(m_global_semaphore);
    ttnn::hd_socket::py_module_types(m_hd_socket);
    ttnn::h2d_stream_service::py_module_types(m_h2d_stream_service);
    ttnn::mesh_socket::py_module_types(m_mesh_socket);
    ttnn::reports::py_module_types(m_reports);
    ttnn::program_descriptors::py_module_types(m_program_descriptors);
    ttnn::tensor_accessor_args::py_module_types(m_tensor_accessor_args);

    // FUNCTIONS / OPERATIONS
    ttnn::tensor::tensor_mem_config_module(m_tensor);
    ttnn::tensor::pytensor_module(m_tensor);
    ttnn::core::py_module(m_core);
    ttnn::graph::py_graph_module(m_graph);

#if defined(TRACY_ENABLE)
    // https://nanobind.readthedocs.io/en/latest/utilities.html
    // breadcrumbs
    // https://github.com/wjakob/nanobind/discussions/302
    // https://github.com/wjakob/nanobind/discussions/671
    // https://nanobind.readthedocs.io/en/latest/api_core.html#parameterized-wrapper-classes
    // https://nanobind.readthedocs.io/en/latest/api_core.html#_CPPv4N8nanobind3sigE
    nb::callable tracy_decorator = nb::module_::import_("tracy.ttnn_profiler_wrapper").attr("callable_decorator");

    tracy_decorator(m_device);
    tracy_decorator(m_tensor);
    tracy_decorator(m_depr_operations);
#endif

    auto m_bfp_utils = mod.def_submodule("bfp_utils", "BFP tile pack/unpack utilities");
    ttnn::bfp_utils::py_module(m_bfp_utils);

    ttnn::types::py_module(m_types);
    ttnn::activation::py_module(m_activation);
    ttnn::cluster::py_cluster_module(m_cluster);
    ttnn::device::py_device_module(m_device);
    ttnn::distributed::py_module(m_multi_device);
    ttnn::events::py_module(m_events);
    ttnn::global_circular_buffer::py_module(m_global_circular_buffer);
    ttnn::global_semaphore::py_module(m_global_semaphore);
    ttnn::hd_socket::py_module(m_hd_socket);
    ttnn::h2d_stream_service::py_module(m_h2d_stream_service);
    ttnn::mesh_socket::py_module(m_mesh_socket);
    ttnn::profiler::py_module(m_profiler);
    ttnn::reports::py_module(m_reports);
    ttnn::tensor_accessor_args::py_module(m_tensor_accessor_args);

    // ttnn operations have to come before the deprecated ones,
    // because ttnn defines additional type bindings.
    // TODO: pull them out of the ttnn::operations::py_module.
    ttnn::operations::py_module(m_operations);
    // tt::operations::primary::py_module(m_primary_ops);

    // CONFIG is a shared mutable global: Python code reads and writes properties
    // via setattr (e.g. manage_config context manager).  We must bind by reference
    // so mutations are visible across C++ and Python.  Suppress the leak warning
    // for this one binding — the object is intentionally static-lifetime.
    nb::set_leak_warnings(false);
    mod.attr("CONFIG") = nb::cast(&ttnn::CONFIG, nb::rv_policy::reference);
    nb::set_leak_warnings(true);
    mod.def(
        "get_python_operation_id",
        []() -> std::uint64_t { return ttnn::CoreIDs::instance().get_python_operation_id(); },
        "Get operation id");
    mod.def(
        "set_python_operation_id",
        [](std::uint64_t id) { ttnn::CoreIDs::instance().set_python_operation_id(id); },
        "Set operation id");
    mod.def(
        "fetch_and_increment_python_operation_id",
        []() -> std::uint64_t { return ttnn::CoreIDs::instance().fetch_and_increment_python_operation_id(); },
        "Increment tensor id and return the previously held id");

    mod.def("get_tensor_id", &tt::tt_metal::Tensor::get_tensor_id_counter, "Get the current tensor ID counter value");
    mod.def(
        "set_tensor_id",
        &tt::tt_metal::Tensor::set_tensor_id_counter,
        nb::arg("id"),
        "Set the tensor ID counter to a specific value");
    mod.def(
        "fetch_and_increment_tensor_id",
        &tt::tt_metal::Tensor::next_tensor_id,
        "Atomically fetch and increment the tensor ID counter");

    mod.def(
        "get_device_operation_id",
        []() -> std::uint64_t { return ttnn::CoreIDs::instance().get_device_operation_id(); },
        "Get device operation id");
    mod.def(
        "set_device_operation_id",
        [](std::uint64_t id) { ttnn::CoreIDs::instance().set_device_operation_id(id); },
        "Set device operation id");
    mod.def(
        "fetch_and_increment_device_operation_id",
        []() -> std::uint64_t { return ttnn::CoreIDs::instance().fetch_and_increment_device_operation_id(); },
        "Increment device operation id and return the previously held id");
}
