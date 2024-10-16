// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "third_party/json/json.hpp"
#include "graph_processor.hpp"
#include "graph_trace_utils.hpp"

#include <vector>

namespace ttnn::graph {

template <class Callable>
auto query_trace(Callable&& callable) {
    GraphProcessor::begin_graph_capture(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH);
    { auto output = callable(); }
    auto json_trace = GraphProcessor::end_graph_capture();
    return json_trace;
}

template <class Callable>
auto query_peak_L1_memory_usage(Callable&& callable) {
    GraphProcessor::begin_graph_capture(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH);
    { auto output = callable(); }
    auto json_trace = GraphProcessor::end_graph_capture();
    return graph::extract_peak_L1_memory_usage(json_trace);
}

template <class Callable>
auto query_output_info(Callable&& callable) {
    GraphProcessor::begin_graph_capture(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH);
    { auto output = callable(); }
    auto json_trace = GraphProcessor::end_graph_capture();
    return graph::extract_output_info(json_trace);
}

}  // namespace ttnn::graph
