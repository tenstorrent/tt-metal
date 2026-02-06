// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/graph/graph_nanobind.hpp"

#include <string>
#include <sstream>
#include <filesystem>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_set.h>
#include <nanobind/stl/filesystem.h>
#include <nlohmann/json.hpp>

#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/graph/graph_consts.hpp"

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

    // Bind PeakMemoryUsagePerCore struct for extract_resource_usage_per_core
    nb::class_<ttnn::graph::PeakMemoryUsagePerCore>(m, "PeakMemoryUsagePerCore")
        .def_ro(
            "peak_cb", &ttnn::graph::PeakMemoryUsagePerCore::peak_cb, "Peak circular buffer usage per core in bytes")
        .def_ro("peak_l1", &ttnn::graph::PeakMemoryUsagePerCore::peak_l1, "Peak L1 buffer usage per core in bytes")
        .def_ro(
            "peak_total",
            &ttnn::graph::PeakMemoryUsagePerCore::peak_total,
            "Peak total memory (CB + L1) per core in bytes")
        .def(
            "__repr__",
            [](const ttnn::graph::PeakMemoryUsagePerCore& usage) {
                std::stringstream ss;
                ss << "PeakMemoryUsagePerCore(peak_cb=" << usage.peak_cb << ", peak_l1=" << usage.peak_l1
                   << ", peak_total=" << usage.peak_total << ")";
                return ss.str();
            })
        .def("__str__", [](const ttnn::graph::PeakMemoryUsagePerCore& usage) {
            std::stringstream ss;
            ss << "Peak Memory Usage Per Core:\n"
               << "  CB:    " << usage.peak_cb << " bytes\n"
               << "  L1:    " << usage.peak_l1 << " bytes\n"
               << "  Total: " << usage.peak_total << " bytes";
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
        R"doc(end_graph_capture() -> list
        Ends graph capture and returns the captured graph trace.

        Returns:
            List of graph nodes, where each node is a dict with keys like
            'node_type', 'counter', 'params', 'connections', etc.
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

    const auto* doc_end_to_file =
        R"doc(end_graph_capture_to_file(report_path) -> list
        Ends graph capture and writes a full report to a JSON file.

        The report file contains: graph trace, device info, and metadata.
        Useful for offline analysis without Python at capture time.

        Args:
            report_path: Path to write the JSON report file

        Returns:
            The captured graph trace (same as end_graph_capture())
    )doc";

    m.def(
        "end_graph_capture_to_file",
        [](const std::filesystem::path& report_path) {
            nlohmann::json json_object = GraphProcessor::end_graph_capture_to_file(report_path);
            auto json_object_str = json_object.dump();
            auto json_module = nb::module_::import_("json");
            return json_module.attr("loads")(json_object_str);
        },
        doc_end_to_file,
        nb::arg("report_path"));

    const auto* doc_get_report =
        R"doc(get_current_report() -> dict
        Gets the current capture's full report (graph + devices + metadata) without ending capture.

        This allows you to inspect the current state of capture while it's still active.

        Returns:
            A dictionary containing:
            - version: Report format version
            - graph: The captured graph trace so far
            - devices: Information about captured devices
            - metadata: Capture metadata (timestamps, etc.)

        Example:
            >>> ttnn.graph.begin_graph_capture()
            >>> # ... run some operations ...
            >>> report = ttnn.graph.get_current_report()
            >>> print(f"Captured {len(report['graph'])} nodes so far")
    )doc";

    m.def(
        "get_current_report",
        [] {
            nlohmann::json json_object = GraphProcessor::get_current_report();
            auto json_object_str = json_object.dump();
            auto json_module = nb::module_::import_("json");
            return json_module.attr("loads")(json_object_str);
        },
        doc_get_report);

    // Expose report version constant
    m.attr("REPORT_VERSION") = kCurrentReportVersion;

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

    m.def(
        "extract_resource_usage_per_core",
        [](const nb::object& py_trace, size_t interleaved_storage_cores) {
            auto json_module = nb::module_::import_("json");
            auto trace_str = std::string{nb::str(json_module.attr("dumps")(py_trace)).c_str()};
            nlohmann::json trace = nlohmann::json::parse(trace_str);
            return extract_resource_usage_per_core(trace, interleaved_storage_cores);
        },
        R"doc(
        Extract resource usage per core from graph trace.

        This function calculates the worst-case peak memory usage per core, accounting for
        how buffers are distributed across cores. This is more accurate than total memory
        usage for devices with distributed memory architectures.

        Args:
            trace: Captured graph trace from end_graph_capture()
            interleaved_storage_cores: Number of cores used for interleaved storage.
                                       Typically grid_size.x * grid_size.y

        Returns:
            PeakMemoryUsagePerCore: Object with three fields:
                - peak_cb: Peak circular buffer usage per core (bytes)
                - peak_l1: Peak L1 buffer usage per core (bytes)
                - peak_total: Peak total memory (CB + L1) per core (bytes)

        Example:
            >>> grid_size = device.compute_with_storage_grid_size()
            >>> cores = grid_size.x * grid_size.y
            >>> usage = ttnn.graph.extract_resource_usage_per_core(graph, cores)
            >>> print(f"Peak per core: {usage.peak_total:,} bytes")
            >>> if usage.peak_total > 256 * 1024:
            ...     print("Warning: Exceeds 256KB L1 per core!")
        )doc",
        nb::arg("trace"),
        nb::arg("interleaved_storage_cores"));

    m.def(
        "enable_stack_traces",
        &GraphProcessor::enable_stack_traces,
        R"doc(enable_stack_traces() -> None

        Enable C++ stack trace capture for graph operations.

        When enabled, each function_start node in the captured graph will include
        a 'stack_trace' field containing the C++ call stack at that point.

        This is useful for debugging to understand where operations are called from.

        Note:
            - Only available on Linux/macOS (uses backtrace())
            - Adds overhead to graph capture - only enable when needed
            - Stack traces show C++ function names (demangled when possible)

        Example:
            >>> ttnn.graph.enable_stack_traces()
            >>> ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            >>> # ... operations ...
            >>> graph = ttnn.graph.end_graph_capture()
            >>> # function_start nodes now have 'stack_trace' field
            >>> ttnn.graph.disable_stack_traces()
        )doc");

    m.def(
        "disable_stack_traces",
        &GraphProcessor::disable_stack_traces,
        R"doc(disable_stack_traces() -> None

        Disable C++ stack trace capture for graph operations.

        Call this after you're done debugging to remove the capture overhead.
        )doc");

    m.def(
        "is_stack_trace_enabled",
        &GraphProcessor::is_stack_trace_enabled,
        R"doc(is_stack_trace_enabled() -> bool

        Check if stack trace capture is currently enabled.

        Returns:
            True if stack traces are being captured, False otherwise.
        )doc");

    m.def(
        "enable_buffer_pages",
        &GraphProcessor::enable_buffer_pages,
        R"doc(enable_buffer_pages() -> None

        Enable detailed buffer page capture for graph reports.

        When enabled, the graph report will include a 'buffer_pages' array with
        detailed per-page information for all L1 buffers:
        - device_id: Device the buffer is on
        - address: Buffer base address
        - core_y, core_x: Core coordinates where the page is stored
        - bank_id: L1 bank ID
        - page_index: Index of this page within the buffer
        - page_address: Actual address of this specific page
        - page_size: Size of each page in bytes
        - buffer_type: Buffer type (0=DRAM, 1=L1, etc.)

        This is useful for understanding memory layout and debugging sharding issues.

        Note:
            - Only captures L1 buffers (not DRAM)
            - Adds overhead - only enable when needed
            - Buffer state is captured at end of graph capture

        Example:
            >>> ttnn.graph.enable_buffer_pages()
            >>> ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            >>> # ... operations ...
            >>> graph = ttnn.graph.end_graph_capture_to_file("report.json")
            >>> # report.json now has 'buffer_pages' array
            >>> ttnn.graph.disable_buffer_pages()
        )doc");

    m.def(
        "disable_buffer_pages",
        &GraphProcessor::disable_buffer_pages,
        R"doc(disable_buffer_pages() -> None

        Disable detailed buffer page capture.

        Call this after you're done with buffer analysis to remove the capture overhead.
        )doc");

    m.def(
        "is_buffer_pages_enabled",
        &GraphProcessor::is_buffer_pages_enabled,
        R"doc(is_buffer_pages_enabled() -> bool

        Check if detailed buffer page capture is currently enabled.

        Returns:
            True if buffer pages are being captured, False otherwise.
        )doc");
}

}  // namespace ttnn::graph
