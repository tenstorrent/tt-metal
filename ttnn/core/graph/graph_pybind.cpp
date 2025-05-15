// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/graph/graph_pybind.hpp"

#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"

#include "pybind11/stl.h"

namespace ttnn::graph {

namespace py = pybind11;
using IGraphProcessor = tt::tt_metal::IGraphProcessor;

void py_graph_module_types(py::module& m) {
    py::enum_<IGraphProcessor::RunMode>(m, "RunMode")
        .value("NORMAL", IGraphProcessor::RunMode::NORMAL)
        .value("NO_DISPATCH", IGraphProcessor::RunMode::NO_DISPATCH);
}

void py_graph_module(py::module& m) {
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
}

}  // namespace ttnn::graph
