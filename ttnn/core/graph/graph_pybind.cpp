// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/graph/graph_pybind.hpp"

#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"

#include "pybind11/stl.h"

#include <sstream>

namespace ttnn::graph {

namespace py = pybind11;
using IGraphProcessor = tt::tt_metal::IGraphProcessor;

void py_graph_module_types(py::module& m) {
    py::enum_<IGraphProcessor::RunMode>(m, "RunMode")
        .value("NORMAL", IGraphProcessor::RunMode::NORMAL)
        .value("NO_DISPATCH", IGraphProcessor::RunMode::NO_DISPATCH);
}

void py_graph_module(py::module& m) {
    // Bind TensorInfo struct for extract_output_info
    py::class_<ttnn::graph::TensorInfo>(m, "TensorInfo")
        .def_readonly("shape", &ttnn::graph::TensorInfo::shape)
        .def_readonly("size", &ttnn::graph::TensorInfo::size)
        .def_readonly("type", &ttnn::graph::TensorInfo::type)
        .def("__repr__", [](const ttnn::graph::TensorInfo& info) {
            std::stringstream ss;
            ss << "TensorInfo(shape=" << info.shape << ", size=" << info.size
               << ", type=" << (info.type == tt::tt_metal::BufferType::DRAM ? "DRAM" : "L1") << ")";
            return ss.str();
        });

    auto doc_begin =
        R"doc(begin_graph_capture()
    )doc";

    m.def(
        "begin_graph_capture",
        &GraphProcessor::begin_graph_capture,
        doc_begin,
        py::arg("run_mode") = IGraphProcessor::RunMode::NORMAL);

    auto doc_end =
        R"doc(end_graph_capture() -> Union[None, bool, int, float, list, dict]
        returns the value captured graph as a json object converted to python object
    )doc";

    m.def(
        "end_graph_capture",
        [] {
            nlohmann::json json_object = GraphProcessor::end_graph_capture();
            auto json_object_str = json_object.dump();
            auto json_module = py::module::import("json");
            return json_module.attr("loads")(json_object_str);
        },
        doc_end);

    m.def(
        "extract_calltrace",
        [](const py::object& py_trace) {
            auto json_module = py::module::import("json");
            std::string trace_str = py::str(json_module.attr("dumps")(py_trace));
            nlohmann::json trace = nlohmann::json::parse(trace_str);
            return extract_calltrace(trace);
        },
        "Extracts calltrace from the graph trace",
        py::arg("trace"));

    m.def(
        "extract_peak_L1_memory_usage",
        [](const py::object& py_trace) {
            auto json_module = py::module::import("json");
            std::string trace_str = py::str(json_module.attr("dumps")(py_trace));
            nlohmann::json trace = nlohmann::json::parse(trace_str);
            return extract_peak_L1_memory_usage(trace);
        },
        "Extracts peak L1 memory usage from the graph trace in bytes",
        py::arg("trace"));

    m.def(
        "count_intermediate_and_output_tensors",
        [](const py::object& py_trace) {
            auto json_module = py::module::import("json");
            std::string trace_str = py::str(json_module.attr("dumps")(py_trace));
            nlohmann::json trace = nlohmann::json::parse(trace_str);
            auto result = count_intermediate_and_output_tensors(trace);
            return py::make_tuple(result.first, result.second);
        },
        "Counts intermediate and output tensors. Returns (intermediate_count, output_count)",
        py::arg("trace"));

    m.def(
        "extract_output_info",
        [](const py::object& py_trace) {
            auto json_module = py::module::import("json");
            std::string trace_str = py::str(json_module.attr("dumps")(py_trace));
            nlohmann::json trace = nlohmann::json::parse(trace_str);
            return extract_output_info(trace);
        },
        "Extracts output tensor information. Returns list of TensorInfo objects",
        py::arg("trace"));

    m.def(
        "extract_circular_buffers_peak_size_per_core",
        [](const py::object& py_trace) {
            auto json_module = py::module::import("json");
            std::string trace_str = py::str(json_module.attr("dumps")(py_trace));
            nlohmann::json trace = nlohmann::json::parse(trace_str);
            return extract_circular_buffers_peak_size_per_core(trace);
        },
        "Extracts peak circular buffer size per core in bytes",
        py::arg("trace"));

    m.def(
        "extract_l1_buffer_allocation_peak_size_per_core",
        [](const py::object& py_trace, size_t interleaved_storage_cores) {
            auto json_module = py::module::import("json");
            std::string trace_str = py::str(json_module.attr("dumps")(py_trace));
            nlohmann::json trace = nlohmann::json::parse(trace_str);
            return extract_l1_buffer_allocation_peak_size_per_core(trace, interleaved_storage_cores);
        },
        "Extracts peak L1 buffer allocation size per core. Requires interleaved_storage_cores "
        "(get from device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y)",
        py::arg("trace"),
        py::arg("interleaved_storage_cores"));

    m.def(
        "extract_output_tensors",
        [](const py::object& py_trace) {
            auto json_module = py::module::import("json");
            std::string trace_str = py::str(json_module.attr("dumps")(py_trace));
            nlohmann::json trace = nlohmann::json::parse(trace_str);
            return extract_output_tensors(trace);
        },
        "Extracts output tensor IDs from the trace. Returns set of tensor IDs",
        py::arg("trace"));
}

}  // namespace ttnn::graph
