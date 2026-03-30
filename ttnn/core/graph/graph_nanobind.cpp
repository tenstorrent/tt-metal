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
#include <tt-metalium/graph_tracking.hpp>

namespace ttnn::graph {

using IGraphProcessor = tt::tt_metal::IGraphProcessor;
using GraphTracker = tt::tt_metal::GraphTracker;

void py_graph_module_types(nb::module_& m) {
    nb::enum_<IGraphProcessor::RunMode>(m, "RunMode", R"doc(
        Run mode for graph capture.

        NORMAL: Operations execute normally on the device while being traced.
                Programs are cached for future reuse.

        NO_DISPATCH: Operations are traced but NOT executed on the device.
                     Buffer allocations are mocked (address=0), and programs are
                     compiled but not dispatched. This mode is useful for:
                     - Measuring memory usage of models that don't fit in device memory
                     - Analyzing operation graphs without hardware execution

                     IMPORTANT: Programs are NOT cached during NO_DISPATCH mode because
                     they contain invalid buffer addresses. This ensures that subsequent
                     runs in NORMAL mode will create fresh programs with valid addresses.
    )doc")
        .value("NORMAL", IGraphProcessor::RunMode::NORMAL, "Execute operations normally while tracing")
        .value(
            "NO_DISPATCH",
            IGraphProcessor::RunMode::NO_DISPATCH,
            "Trace operations without executing on device (no program caching)");
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
        R"doc(
        Begin capturing a graph of operations.

        This function starts recording all ttnn operations into a graph that can be
        analyzed for memory usage, operation dependencies, and other metrics.

        Args:
            run_mode: Determines how operations are executed during capture.
                - RunMode.NORMAL (default): Operations execute on the device normally.
                  Programs are compiled and cached for reuse.
                - RunMode.NO_DISPATCH: Operations are traced but NOT executed on device.
                  Buffer allocations are mocked and programs are not dispatched.
                  This allows profiling models that exceed device memory.

                  NOTE: In NO_DISPATCH mode, programs are intentionally NOT cached
                  because they contain invalid buffer addresses (address=0). This
                  ensures that when you later run in NORMAL mode, fresh programs
                  with valid addresses will be created.

        Example:
            >>> # Profile memory without execution
            >>> ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)
            >>> output = model(input_tensor)
            >>> trace = ttnn.graph.end_graph_capture()
            >>> peak_memory = ttnn.graph.extract_peak_L1_memory_usage(trace)

            >>> # Normal execution with tracing
            >>> ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            >>> output = model(input_tensor)
            >>> trace = ttnn.graph.end_graph_capture()
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
        R"doc(end_graph_capture_to_file(report_path) -> str
        Ends graph capture and writes a full report to a JSON file.

        The report file contains: graph trace, device info, and metadata.
        Useful for offline analysis without Python at capture time.

        Args:
            report_path: Path to write the JSON report file

        Returns:
            The captured graph trace as a JSON string
    )doc";

    m.def(
        "end_graph_capture_to_file",
        [](const std::filesystem::path& report_path) {
            nlohmann::json json_object = GraphProcessor::end_graph_capture_to_file(report_path);
            return json_object.dump();
        },
        doc_end_to_file,
        nb::arg("report_path"));

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
        "is_graph_capture_active",
        [] { return GraphTracker::instance().is_enabled(); },
        R"doc(is_graph_capture_active() -> bool

        Check if a graph capture session is currently active.

        Returns:
            True if begin_graph_capture() was called and capture is in progress.
        )doc");

    m.def(
        "track_function_start",
        [](const std::string& function_name) {
            GraphTracker::instance().track_function_start(std::string_view(function_name));
        },
        R"doc(track_function_start(function_name: str) -> None

        Emit a function_start node into the active graph capture.

        This is used by Python-level operation decorators to wrap high-level
        operations (e.g. ttnn.fold, ttnn.conv2d) so that the graph trace
        correctly nests their internal C++ sub-operations.

        Args:
            function_name: The operation name (e.g. "ttnn.fold")
        )doc",
        nb::arg("function_name"));

    m.def(
        "track_function_end",
        [] { GraphTracker::instance().track_function_end(); },
        R"doc(track_function_end() -> None

        Emit a function_end node into the active graph capture.

        Must be paired with a prior track_function_start() call.
        )doc");

    m.def(
        "enable_detailed_buffer_tracing",
        &GraphProcessor::enable_detailed_buffer_tracing,
        R"doc(enable_detailed_buffer_tracing() -> None

        Enable detailed per-page buffer tracing for graph reports.
        )doc");

    m.def(
        "disable_detailed_buffer_tracing",
        &GraphProcessor::disable_detailed_buffer_tracing,
        R"doc(disable_detailed_buffer_tracing() -> None

        Disable detailed per-page buffer tracing.
        )doc");

    m.def(
        "is_detailed_buffer_tracing_enabled",
        &GraphProcessor::is_detailed_buffer_tracing_enabled,
        R"doc(is_detailed_buffer_tracing_enabled() -> bool

        Check if detailed per-page buffer tracing is currently enabled.
        )doc");
}

}  // namespace ttnn::graph
