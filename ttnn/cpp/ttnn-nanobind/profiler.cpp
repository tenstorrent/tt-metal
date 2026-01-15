// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "profiler.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>
#include <nanobind/operators.h>

#include "tools/profiler/op_profiler.hpp"
#include "tt-metalium/experimental/profiler.hpp"

namespace ttnn::profiler {

namespace {

namespace ttm = tt::tt_metal::experimental;

nb::dict convert_sets_to_lists(const std::map<tt::ChipId, std::set<ttm::ProgramAnalysisData>>& perf_data) {
    nb::dict out;
    for (const auto& [chip_id, program_set] : perf_data) {
        nb::list programs;
        for (const auto& program_data : program_set) {
            programs.append(nb::cast(program_data));
        }
        out[nb::int_(chip_id)] = programs;
    }
    return out;
}

nb::dict convert_kernel_duration_summaries_to_dict(const std::map<tt::ChipId, ttm::KernelDurationSummary>& summaries) {
    nb::dict out;
    for (const auto& [chip_id, summary] : summaries) {
        out[nb::int_(chip_id)] = nb::cast(summary);
    }
    return out;
}

void ProfilerModule(nb::module_& mod) {
    nb::class_<ttm::ProgramExecutionUID>(mod, "ProgramExecutionUID")
        .def(nb::init<>())
        .def_rw("runtime_id", &ttm::ProgramExecutionUID::runtime_id)
        .def_rw("trace_id", &ttm::ProgramExecutionUID::trace_id)
        .def_rw("trace_id_counter", &ttm::ProgramExecutionUID::trace_id_counter)
        .def("__eq__", &ttm::ProgramExecutionUID::operator==)
        .def("__lt__", &ttm::ProgramExecutionUID::operator<)
        .def("__repr__", [](const ttm::ProgramExecutionUID& uid) {
            return fmt::format(
                "ProgramExecutionUID(runtime_id={}, trace_id={}, trace_id_counter={})",
                uid.runtime_id,
                uid.trace_id,
                uid.trace_id_counter);
        });

    nb::class_<ttm::ProgramSingleAnalysisResult>(mod, "ProgramSingleAnalysisResult")
        .def(nb::init<>())
        .def_rw("start_timestamp", &ttm::ProgramSingleAnalysisResult::start_timestamp)
        .def_rw("end_timestamp", &ttm::ProgramSingleAnalysisResult::end_timestamp)
        .def_rw("duration", &ttm::ProgramSingleAnalysisResult::duration)
        .def("__eq__", &ttm::ProgramSingleAnalysisResult::operator==)
        .def("__ne__", &ttm::ProgramSingleAnalysisResult::operator!=)
        .def("__lt__", &ttm::ProgramSingleAnalysisResult::operator<)
        .def("__repr__", [](const ttm::ProgramSingleAnalysisResult& result) {
            return fmt::format(
                "ProgramSingleAnalysisResult(start_timestamp={}, end_timestamp={}, duration={})",
                result.start_timestamp,
                result.end_timestamp,
                result.duration);
        });

    nb::class_<ttm::ProgramAnalysisData>(mod, "ProgramAnalysisData")
        .def(nb::init<>())
        .def_rw("program_execution_uid", &ttm::ProgramAnalysisData::program_execution_uid)
        .def_rw("program_analyses_results", &ttm::ProgramAnalysisData::program_analyses_results)
        .def_rw("core_count", &ttm::ProgramAnalysisData::core_count)
        .def_rw("num_available_cores", &ttm::ProgramAnalysisData::num_available_cores)
        .def("__eq__", &ttm::ProgramAnalysisData::operator==)
        .def("__lt__", &ttm::ProgramAnalysisData::operator<)
        .def("__repr__", [](const ttm::ProgramAnalysisData& data) {
            const std::string uid_repr = nb::cast<std::string>(nb::repr(nb::cast(data.program_execution_uid)));
            return fmt::format(
                "ProgramAnalysisData(program_execution_uid={}, program_analyses_results_size={}, core_count={}, "
                "num_available_cores={})",
                uid_repr,
                data.program_analyses_results.size(),
                data.core_count,
                data.num_available_cores);
        });

    nb::class_<ttm::DurationHistogram>(mod, "DurationHistogram")
        .def(nb::init<>())
        .def_rw("min_ns", &ttm::DurationHistogram::min_ns)
        .def_rw("max_ns", &ttm::DurationHistogram::max_ns)
        .def_rw("num_buckets", &ttm::DurationHistogram::num_buckets)
        .def_rw("bucket_edges_ns", &ttm::DurationHistogram::bucket_edges_ns)
        .def_rw("bucket_counts", &ttm::DurationHistogram::bucket_counts)
        .def_rw("underflow", &ttm::DurationHistogram::underflow)
        .def_rw("overflow", &ttm::DurationHistogram::overflow);

    nb::class_<ttm::KernelDurationSummary>(mod, "KernelDurationSummary")
        .def(nb::init<>())
        .def_rw("count", &ttm::KernelDurationSummary::count)
        .def_rw("min_ns", &ttm::KernelDurationSummary::min_ns)
        .def_rw("max_ns", &ttm::KernelDurationSummary::max_ns)
        .def_rw("avg_ns", &ttm::KernelDurationSummary::avg_ns)
        .def_rw("histogram", &ttm::KernelDurationSummary::histogram);
    mod.def(
        "start_tracy_zone",
        &tt::tt_metal::op_profiler::start_tracy_zone,
        nb::arg("source"),
        nb::arg("functName"),
        nb::arg("lineNum"),
        nb::arg("color") = 0,
        R"doc(
        Stop profiling op with tracy.
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                                    | Data type             | Valid range | Required |
        +==================+================================================+=======================+=============+==========+
        | source           | Source file for the zone                       | string                |             | Yes      |
        | functName        | Function of the zone                           | string                |             | Yes      |
        | lineNum          | Line number of the zone marker                 | int                   |             | Yes      |
        | color            | Zone color                                     | int                   |             | No       |
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
    )doc");

    mod.def(
        "stop_tracy_zone",
        &tt::tt_metal::op_profiler::stop_tracy_zone,
        nb::arg("name") = "",
        nb::arg("color") = 0,
        R"doc(
        Stop profiling op with tracy.
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                                    | Data type             | Valid range | Required |
        +==================+================================================+=======================+=============+==========+
        | name             | Replace name for the zone                          | string                |             | No       |
        | color            | Replace zone color                             | int                   |             | No       |
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
    )doc");

    mod.def(
        "tracy_message",
        &tt::tt_metal::op_profiler::tracy_message,
        nb::arg("message"),
        nb::arg("color") = 0xf0f8ff,
        R"doc(
        Emit a message signpost into the tracy profile.
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                                    | Data type             | Valid range | Required |
        +==================+================================================+=======================+=============+==========+
        | message          | Message description for this signpost.         | string                |             | Yes      |
        | color            | Zone color                                     | int                   |             | No       |
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
    )doc");

    mod.def(
        "tracy_frame",
        &tt::tt_metal::op_profiler::tracy_frame,
        R"doc(
        Emit a tracy frame signpost.
    )doc");

    mod.def(
        "get_latest_programs_perf_data",
        []() { return convert_sets_to_lists(tt::tt_metal::experimental::GetLatestProgramsPerfData()); },
        R"doc(
        Get performance results for all programs that were read in the most recent call to `ttnn.ReadDeviceProfiler()`.
        Returns a dictionary mapping chip IDs to lists of ProgramAnalysisData objects. The list contains only the latest entry for each program.
        TT_METAL_DEVICE_PROFILER=1, TT_METAL_PROFILER_MID_RUN_DUMP=1, and TT_METAL_PROFILER_CPP_POST_PROCESS=1 environment variables must be set.
        Returns an empty dictionary if the environment variables are not set.
    )doc");

    mod.def(
        "get_all_programs_perf_data",
        []() { return convert_sets_to_lists(tt::tt_metal::experimental::GetAllProgramsPerfData()); },
        R"doc(
        Get performance results for all programs that have been read so far across all calls to `ttnn.ReadDeviceProfiler()`.
        Returns a dictionary mapping chip IDs to lists of ProgramAnalysisData objects. The list contains all entries for each program.
        TT_METAL_DEVICE_PROFILER=1, TT_METAL_PROFILER_MID_RUN_DUMP=1, and TT_METAL_PROFILER_CPP_POST_PROCESS=1 environment variables must be set.
        Returns an empty dictionary if the environment variables are not set.
    )doc");

    mod.def(
        "get_latest_kernel_duration_summary",
        []() {
            return convert_kernel_duration_summaries_to_dict(
                tt::tt_metal::experimental::GetLatestKernelDurationSummary());
        },
        R"doc(
        Get a summary (min/max/avg + histogram) of DEVICE KERNEL duration for the latest captured program set.
        Returns a dictionary mapping chip IDs to KernelDurationSummary objects.
        TT_METAL_DEVICE_PROFILER=1, TT_METAL_PROFILER_MID_RUN_DUMP=1, and TT_METAL_PROFILER_CPP_POST_PROCESS=1 environment variables must be set.
        Returns an empty dictionary if the environment variables are not set.
    )doc");

    mod.def(
        "get_all_kernel_duration_summary",
        []() {
            return convert_kernel_duration_summaries_to_dict(tt::tt_metal::experimental::GetAllKernelDurationSummary());
        },
        R"doc(
        Get a summary (min/max/avg + histogram) of DEVICE KERNEL duration across all captured program sets so far.
        Returns a dictionary mapping chip IDs to KernelDurationSummary objects.
        TT_METAL_DEVICE_PROFILER=1, TT_METAL_PROFILER_MID_RUN_DUMP=1, and TT_METAL_PROFILER_CPP_POST_PROCESS=1 environment variables must be set.
        Returns an empty dictionary if the environment variables are not set.
    )doc");
}

}  // namespace

void py_module(nb::module_& mod) { ProfilerModule(mod); }

}  // namespace ttnn::profiler
