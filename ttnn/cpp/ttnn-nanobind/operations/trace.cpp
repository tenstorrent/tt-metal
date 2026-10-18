// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "trace.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/unordered_set.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/experimental/allocation_context.hpp>
#include <tt-metalium/experimental/trace_allocation_tracker.hpp>

#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/trace.hpp"

namespace ttnn::operations::trace {

namespace tracker = tt::tt_metal::distributed::trace_allocation_tracker;

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
}

void py_module(nb::module_& mod) {
    // Read the process-wide tracker configuration from Metal's cached RunTimeOptions snapshot. Python must not parse
    // these environment variables independently because the two layers must always agree.
    mod.def("trace_allocation_tracking_enabled", []() { return tt::tt_metal::trace_allocation_tracking_enabled(); });
    mod.def(
        "trace_allocation_diagnostics_enabled", []() { return tt::tt_metal::trace_allocation_diagnostics_enabled(); });

    mod.def(
        "begin_trace_capture",
        [](MeshDevice* device, std::optional<ttnn::QueueId> cq_id) {
            return ttnn::operations::trace::begin_trace_capture(device, cq_id);
        },
        nb::sig("def begin_trace_capture(mesh_device: ttnn.MeshDevice, \\*, cq_id: Optional[ttnn.QueueId] = None) -> "
                "ttnn.MeshTraceId"),
        nb::arg("mesh_device"),
        nb::kw_only(),
        nb::arg("cq_id") = nb::none(),
        nb::call_guard<nb::gil_scoped_release>(),
        R"doc(
            Begins recording commands for a trace on the mesh device.

            The trace is created under the currently active sub-device manager. The returned trace ID is scoped to
            that manager. If another manager is loaded later, reload the original manager before ending, executing,
            or releasing this trace.

            Args:
                mesh_device (ttnn.MeshDevice): Mesh device on which to capture the trace.

            Keyword Args:
                cq_id (Optional[ttnn.QueueId]): Command queue to capture. Defaults to the current command queue.

            Returns:
                ttnn.MeshTraceId: ID of the trace under the active sub-device manager.
        )doc");

    mod.def(
        "end_trace_capture",
        [](MeshDevice* device, MeshTraceId trace_id, std::optional<ttnn::QueueId> cq_id) {
            ttnn::operations::trace::end_trace_capture(device, trace_id, cq_id);
        },
        nb::sig("def end_trace_capture(mesh_device: ttnn.MeshDevice, trace_id: ttnn.MeshTraceId, \\*, "
                "cq_id: Optional[ttnn.QueueId] = None) -> None"),
        nb::arg("mesh_device"),
        nb::arg("trace_id"),
        nb::kw_only(),
        nb::arg("cq_id") = nb::none(),
        nb::call_guard<nb::gil_scoped_release>(),
        R"doc(
            Ends recording commands for a trace.

            ``trace_id`` is interpreted under the currently active sub-device manager. The same manager that was
            active when capture began must be active for this call.

            Args:
                mesh_device (ttnn.MeshDevice): Mesh device on which the trace was captured.
                trace_id (ttnn.MeshTraceId): ID of the trace under the active sub-device manager.

            Keyword Args:
                cq_id (Optional[ttnn.QueueId]): Command queue being captured. Defaults to the current command queue.
        )doc");

    mod.def(
        "execute_trace",
        [](MeshDevice* device, MeshTraceId trace_id, std::optional<QueueId> cq_id, bool blocking) {
            ttnn::operations::trace::execute_trace(device, trace_id, cq_id, blocking);
        },
        nb::sig("def execute_trace(mesh_device: ttnn.MeshDevice, trace_id: ttnn.MeshTraceId, \\*, "
                "cq_id: Optional[ttnn.QueueId] = None, blocking: bool = True) -> None"),
        nb::arg("mesh_device"),
        nb::arg("trace_id"),
        nb::kw_only(),
        nb::arg("cq_id") = nb::none(),
        nb::arg("blocking") = true,
        nb::call_guard<nb::gil_scoped_release>(),
        R"doc(
            Replays a captured trace.

            ``trace_id`` is interpreted under the currently active sub-device manager. The manager under which the
            trace was captured must be active for this call.

            Args:
                mesh_device (ttnn.MeshDevice): Mesh device on which the trace was captured.
                trace_id (ttnn.MeshTraceId): ID of the trace under the active sub-device manager.

            Keyword Args:
                cq_id (Optional[ttnn.QueueId]): Command queue on which to replay. Defaults to the current command queue.
                blocking (bool): Whether to wait for trace execution to complete. Defaults to ``True``.
        )doc");

    mod.def(
        "release_trace",
        [](MeshDevice* device, MeshTraceId trace_id) { ttnn::operations::trace::release_trace(device, trace_id); },
        nb::arg("mesh_device"),
        nb::arg("trace_id"),
        nb::call_guard<nb::gil_scoped_release>(),
        R"doc(
            Releases a captured trace and its storage.

            ``trace_id`` is interpreted under the currently active sub-device manager. The manager under which the
            trace was captured must be active for this call.

            Args:
                mesh_device (ttnn.MeshDevice): Mesh device on which the trace was captured.
                trace_id (ttnn.MeshTraceId): ID of the trace under the active sub-device manager.
        )doc");

    // Unsafe allocation tracking
    mod.def(
        "get_unsafe_tracked_ids",
        [](MeshDevice* device, MeshTraceId trace_id) { return tracker::get_unsafe_tracked_ids(device, trace_id); },
        nb::arg("mesh_device"),
        nb::arg("trace_id"));
    mod.def(
        "remove_unsafe_tracked_id",
        [](MeshDevice* device, size_t buffer_unique_id) {
            tracker::remove_unsafe_tracked_id(device, buffer_unique_id);
        },
        nb::arg("mesh_device"),
        nb::arg("buffer_unique_id"));
    mod.def("drain_pending_traceback_ids", []() { return tracker::drain_pending_traceback_ids(); });
    mod.def("get_all_unsafe_tracked_ids", []() { return tracker::get_all_unsafe_tracked_ids(); });
    mod.def(
        "push_corruptible_allocation_scope",
        [](MeshDevice* device) { tracker::push_corruptible_allocation_scope(device); },
        nb::arg("mesh_device"));
    mod.def(
        "pop_corruptible_allocation_scope",
        [](MeshDevice* device) { tracker::pop_corruptible_allocation_scope(device); },
        nb::arg("mesh_device"));

    // Allocation context stack
    mod.def(
        "push_allocation_context",
        [](const std::string& ctx) { tt::tt_metal::push_allocation_context(ctx); },
        nb::arg("context"));
    mod.def("pop_allocation_context", []() { tt::tt_metal::pop_allocation_context(); });
}

}  // namespace ttnn::operations::trace
