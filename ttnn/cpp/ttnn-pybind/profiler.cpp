// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "profiler.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include <string>
#include <unordered_map>

#include <fmt/format.h>

#include "tools/profiler/op_profiler.hpp"
#include "tt-metalium/experimental/profiler.hpp"

namespace ttnn::profiler {

namespace {

namespace ttm = tt::tt_metal::experimental;

py::dict to_py_dict(const ttm::ProgramExecutionUID& uid) {
    py::dict d;
    d["runtime_id"] = uid.runtime_id;
    d["trace_id"] = uid.trace_id;
    d["trace_id_counter"] = uid.trace_id_counter;
    return d;
}

py::dict to_py_dict(const ttm::ProgramSingleAnalysisResult& result) {
    py::dict d;
    d["start_timestamp"] = result.start_timestamp;
    d["end_timestamp"] = result.end_timestamp;
    d["duration"] = result.duration;
    return d;
}

py::dict to_py_dict(const ttm::ProgramAnalysisData& data) {
    py::dict d;
    d["program_execution_uid"] = to_py_dict(data.program_execution_uid);

    py::dict analyses;
    for (const auto& [name, single_result] : data.program_analyses_results) {
        analyses[py::str(name)] = to_py_dict(single_result);
    }
    d["program_analyses_results"] = std::move(analyses);

    return d;
}

py::dict convert_programs_perf_data(const std::map<tt::ChipId, std::set<ttm::ProgramAnalysisData>>& perf_data) {
    py::dict out;
    for (const auto& [chip_id, program_set] : perf_data) {
        py::list programs;
        for (const auto& program_data : program_set) {
            programs.append(to_py_dict(program_data));
        }
        out[py::int_(chip_id)] = std::move(programs);
    }
    return out;
}

void ProfilerModule(py::module& m_profiler) {
    py::class_<ttm::ProgramExecutionUID>(m_profiler, "ProgramExecutionUID")
        .def(py::init<>())
        .def_readwrite("runtime_id", &ttm::ProgramExecutionUID::runtime_id)
        .def_readwrite("trace_id", &ttm::ProgramExecutionUID::trace_id)
        .def_readwrite("trace_id_counter", &ttm::ProgramExecutionUID::trace_id_counter)
        .def("__eq__", &ttm::ProgramExecutionUID::operator==)
        .def("__lt__", &ttm::ProgramExecutionUID::operator<)
        .def("__repr__", [](const ttm::ProgramExecutionUID& uid) {
            return fmt::format(
                "ProgramExecutionUID(runtime_id={}, trace_id={}, trace_id_counter={})",
                uid.runtime_id,
                uid.trace_id,
                uid.trace_id_counter);
        });

    py::class_<ttm::ProgramSingleAnalysisResult>(m_profiler, "ProgramSingleAnalysisResult")
        .def(py::init<>())
        .def_readwrite("start_timestamp", &ttm::ProgramSingleAnalysisResult::start_timestamp)
        .def_readwrite("end_timestamp", &ttm::ProgramSingleAnalysisResult::end_timestamp)
        .def_readwrite("duration", &ttm::ProgramSingleAnalysisResult::duration)
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

    py::class_<ttm::ProgramAnalysisData>(m_profiler, "ProgramAnalysisData")
        .def(py::init<>())
        .def_readwrite("program_execution_uid", &ttm::ProgramAnalysisData::program_execution_uid)
        .def_readwrite("program_analyses_results", &ttm::ProgramAnalysisData::program_analyses_results)
        .def("__eq__", &ttm::ProgramAnalysisData::operator==)
        .def("__lt__", &ttm::ProgramAnalysisData::operator<)
        .def("__repr__", [](const ttm::ProgramAnalysisData& data) {
            return fmt::format(
                "ProgramAnalysisData(program_execution_uid={}, program_analyses_results_size={})",
                py::repr(py::cast(data.program_execution_uid)).cast<std::string>(),
                data.program_analyses_results.size());
        });

    m_profiler.def(
        "start_tracy_zone",
        &tt::tt_metal::op_profiler::start_tracy_zone,
        py::arg("source"),
        py::arg("functName"),
        py::arg("lineNum"),
        py::arg("color") = 0,
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

    m_profiler.def(
        "stop_tracy_zone",
        &tt::tt_metal::op_profiler::stop_tracy_zone,
        py::arg("name") = "",
        py::arg("color") = 0,
        R"doc(
        Stop profiling op with tracy.
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                                    | Data type             | Valid range | Required |
        +==================+================================================+=======================+=============+==========+
        | name             | Replace name for the zone                          | string                |             | No       |
        | color            | Replace zone color                             | int                   |             | No       |
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
    )doc");

    m_profiler.def(
        "tracy_message",
        &tt::tt_metal::op_profiler::tracy_message,
        py::arg("message"),
        py::arg("color") = 0xf0f8ff,
        R"doc(
        Emit a message signpost into the tracy profile.
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                                    | Data type             | Valid range | Required |
        +==================+================================================+=======================+=============+==========+
        | message          | Message description for this signpost.         | string                |             | Yes      |
        | color            | Zone color                                     | int                   |             | No       |
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
    )doc");

    m_profiler.def(
        "GetLatestProgramsPerfData",
        []() { return convert_programs_perf_data(tt::tt_metal::experimental::GetLatestProgramsPerfData()); },
        R"doc(
        Get performance results for all programs that were read in the most recent call to `ttnn.ReadDeviceProfile()`.
    )doc");

    m_profiler.def(
        "GetAllProgramsPerfData",
        []() { return convert_programs_perf_data(tt::tt_metal::experimental::GetAllProgramsPerfData()); },
        R"doc(
        Get performance results for all programs that have been read so far across all calls to `ttnn.ReadDeviceProfile()`.
    )doc");
}

}  // namespace

void py_module(py::module& module) { ProfilerModule(module); }

}  // namespace ttnn::profiler
