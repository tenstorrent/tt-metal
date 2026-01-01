// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/graph/graph_nanobind.hpp"

#include <string>
#include <sstream>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_set.h>
#include <nlohmann/json.hpp>

#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"

namespace ttnn::graph {

using IGraphProcessor = tt::tt_metal::IGraphProcessor;

void py_graph_module_types(nb::module_& m) {
    nb::enum_<IGraphProcessor::RunMode>(m, "RunMode")
        .value("NORMAL", IGraphProcessor::RunMode::NORMAL)
        .value("NO_DISPATCH", IGraphProcessor::RunMode::NO_DISPATCH);
}

void py_graph_module(nb::module_& m) {
    // Bind TensorInfo struct for extract_output_info
    nb::class_<ttnn::graph::TensorInfo>(m, "TensorInfo")
        .def_ro("shape", &ttnn::graph::TensorInfo::shape)
        .def_ro("size", &ttnn::graph::TensorInfo::size)
        .def_ro("type", &ttnn::graph::TensorInfo::type)
        .def("__repr__", [](const ttnn::graph::TensorInfo& info) {
            std::stringstream ss;
            std::string type_str;
            switch (info.type) {
                case tt::tt_metal::BufferType::DRAM:
                    type_str = "DRAM";
                    break;
                case tt::tt_metal::BufferType::L1:
                    type_str = "L1";
                    break;
                // Add more cases here as needed for other BufferType values
                default:
                    type_str = "UNKNOWN";
                    break;
            }
            ss << "TensorInfo(shape=" << info.shape << ", size=" << info.size
               << ", type=" << type_str << ")";
            return ss.str();
        });

    const auto* doc_begin =
        R"doc(begin_graph_capture()
    )doc";

    m.def(
        "begin_graph_capture",
        &GraphProcessor::begin_graph_capture,
        doc_begin,
        nb::arg("run_mode") = IGraphProcessor::RunMode::NORMAL);

    const auto* doc_end =
        R"doc(end_graph_capture() -> Union[None, bool, int, float, list, dict]
        returns the value captured graph as a json object converted to python object
    )doc";

    m.def(
        "end_graph_capture",
        [] {
            nlohmann::json json_object = GraphProcessor::end_graph_capture();
            auto json_object_str = json_object.dump();
            auto json_module = nb::module_::import_("json");
            return json_module.attr("loads")(json_object_str);
        },
        doc_end);

    m.def(
        "extract_calltrace",
        [](const nb::object& py_trace) {
            auto json_module = nb::module_::import_("json");
            auto trace_str = std::string{nb::str(json_module.attr("dumps")(py_trace)).c_str()};
            nlohmann::json trace = nlohmann::json::parse(trace_str);
            return extract_calltrace(trace);
        },
        "Extracts calltrace from the graph trace",
        nb::arg("trace"));

    m.def(
        "extract_levelized_graph",
        [](const nb::object& py_trace, size_t max_level) {
            auto json_module = nb::module_::import_("json");
            auto trace_str = std::string{nb::str(json_module.attr("dumps")(py_trace)).c_str()};
            nlohmann::json trace = nlohmann::json::parse(trace_str);
            nlohmann::json levelized_graph = extract_levelized_graph(trace, max_level);
            auto levelized_graph_str = levelized_graph.dump();
            return json_module.attr("loads")(levelized_graph_str);
        },
        "Extracts levelized graph from the graph trace",
        nb::arg("trace"),
        nb::arg("max_level") = 1);

    m.def(
        "extract_peak_L1_memory_usage",
        [](const nb::object& py_trace) {
            auto json_module = nb::module_::import_("json");
            auto trace_str = std::string{nb::str(json_module.attr("dumps")(py_trace)).c_str()};
            nlohmann::json trace = nlohmann::json::parse(trace_str);
            return extract_peak_L1_memory_usage(trace);
        },
        "Extracts peak L1 memory usage from the graph trace in bytes",
        nb::arg("trace"));

    m.def(
        "count_intermediate_and_output_tensors",
        [](const nb::object& py_trace) {
            auto json_module = nb::module_::import_("json");
            auto trace_str = std::string{nb::str(json_module.attr("dumps")(py_trace)).c_str()};
            nlohmann::json trace = nlohmann::json::parse(trace_str);
            auto result = count_intermediate_and_output_tensors(trace);
            return std::make_tuple(result.first, result.second);
        },
        "Counts intermediate and output tensors. Returns (intermediate_count, output_count)",
        nb::arg("trace"));

    m.def(
        "extract_output_info",
        [](const nb::object& py_trace) {
            auto json_module = nb::module_::import_("json");
            auto trace_str = std::string{nb::str(json_module.attr("dumps")(py_trace)).c_str()};
            nlohmann::json trace = nlohmann::json::parse(trace_str);
            return extract_output_info(trace);
        },
        "Extracts output tensor information. Returns list of TensorInfo objects",
        nb::arg("trace"));

    m.def(
        "extract_output_tensors",
        [](const nb::object& py_trace) {
            auto json_module = nb::module_::import_("json");
            auto trace_str = std::string{nb::str(json_module.attr("dumps")(py_trace)).c_str()};
            nlohmann::json trace = nlohmann::json::parse(trace_str);
            return extract_output_tensors(trace);
        },
        "Extracts output tensor IDs from the trace. Returns set of tensor IDs",
        nb::arg("trace"));
}

}  // namespace ttnn::graph
