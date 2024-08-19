// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "graph_processor.hpp"
#include "graph_processor_pybind.hpp"

namespace ttnn {
    namespace py = pybind11;
    using IGraphProcessor = tt::tt_metal::IGraphProcessor;
    void py_graph_module(py::module& m) {
        py::enum_<IGraphProcessor::RunMode>(m, "GraphRunMode")
        .value("FAKE", IGraphProcessor::RunMode::FAKE)
        .value("REAL", IGraphProcessor::RunMode::REAL);
        auto doc_begin =
            R"doc(begin_graph_capture()
        )doc";
        auto doc_end =
            R"doc(end_graph_capture() -> Union[None, bool, int, float, list, dict]
            returns the value captured graph as a json object converted to python object
        )doc";

        m.def("begin_graph_capture", &GraphProcessor::begin_graph_capture, doc_begin,
        py::arg("run_mode") = IGraphProcessor::RunMode::REAL
        );
        m.def(
            "end_graph_capture",
            []{
                nlohmann::json json_object = GraphProcessor::end_graph_capture();
                auto json_object_str = json_object.dump();
                auto json_module = py::module::import("json");
                return json_module.attr("loads")(json_object_str);
            },
            doc_end);
    }
}
