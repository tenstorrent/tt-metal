// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "trace.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/optional.h>

#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/trace.hpp"

namespace ttnn::operations::trace {

void py_module_types(nb::module_& mod) {
    nb::class_<ttnn::MeshTraceId>(mod, "MeshTraceId")
        .def(nb::init<uint32_t>())
        .def("__int__", [](const ttnn::MeshTraceId& self) { return static_cast<int>(*self); })
        .def(
            "__repr__",
            [](const ttnn::MeshTraceId& self) {
                return "MeshTraceId(" + std::to_string(static_cast<int>(*self)) + ")";
            })
        .def(nb::self == nb::self);

    nb::enum_<ttnn::TracePolicy>(mod, "TracePolicy", nb::is_flag(), R"doc(
        Trace-capture policy. Pass to begin_trace_capture(..., policy=...) to enforce
        additional checks over a capture region. Flags compose with ``|``.

        - NONE (default): no policies enforced.
        - REQUIRE_STABLE_CACHE: fatal on ops whose program-cache key depends on live device
          state (e.g. matmul without an explicit program_config, which queries free L1 to pick
          blocking parameters). Set this when the capture must be reproducible across replays.
    )doc")
        .value("NONE", ttnn::TracePolicy::NONE)
        .value("REQUIRE_STABLE_CACHE", ttnn::TracePolicy::REQUIRE_STABLE_CACHE);
}

void py_module(nb::module_& mod) {
    mod.def(
        "begin_trace_capture",
        [](MeshDevice* device, std::optional<ttnn::QueueId> cq_id, ttnn::TracePolicy policy) {
            return ttnn::operations::trace::begin_trace_capture(device, cq_id, policy);
        },
        nb::arg("mesh_device"),
        nb::kw_only(),
        nb::arg("cq_id") = nb::none(),
        nb::arg("policy") = ttnn::TracePolicy::NONE,
        nb::call_guard<nb::gil_scoped_release>());

    mod.def(
        "end_trace_capture",
        [](MeshDevice* device, MeshTraceId trace_id, std::optional<ttnn::QueueId> cq_id) {
            ttnn::operations::trace::end_trace_capture(device, trace_id, cq_id);
        },
        nb::arg("mesh_device"),
        nb::arg("trace_id"),
        nb::kw_only(),
        nb::arg("cq_id") = nb::none(),
        nb::call_guard<nb::gil_scoped_release>());

    mod.def(
        "execute_trace",
        [](MeshDevice* device, MeshTraceId trace_id, std::optional<QueueId> cq_id, bool blocking) {
            ttnn::operations::trace::execute_trace(device, trace_id, cq_id, blocking);
        },
        nb::arg("mesh_device"),
        nb::arg("trace_id"),
        nb::kw_only(),
        nb::arg("cq_id") = nb::none(),
        nb::arg("blocking") = true,
        nb::call_guard<nb::gil_scoped_release>());

    mod.def(
        "release_trace",
        [](MeshDevice* device, MeshTraceId trace_id) { ttnn::operations::trace::release_trace(device, trace_id); },
        nb::arg("mesh_device"),
        nb::arg("trace_id"),
        nb::call_guard<nb::gil_scoped_release>());
}

} // namespace ttnn::operations::trace
