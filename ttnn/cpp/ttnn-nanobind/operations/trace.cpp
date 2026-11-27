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
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <tt-metalium/experimental/trace_store.hpp>

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
        "set_trace_capture_hold",
        [](MeshDevice* device, bool enabled) { device->set_trace_capture_hold(enabled); },
        nb::arg("mesh_device"),
        nb::arg("enabled"),
        R"doc(
            While enabled, a trace capture begun on this mesh device keeps every DRAM buffer freed during the
            capture reserved until that trace is released. The trace rewrites its intermediates' addresses on
            every replay; the hold stops a buffer allocated after capture from landing on one of them.
            Costs the capture's intermediates without reuse between them, held for the trace's lifetime.
        )doc");

    mod.def(
        "trace_capture_hold_bytes",
        [](MeshDevice* device, MeshTraceId trace_id) { return device->trace_capture_hold_bytes(trace_id); },
        nb::arg("mesh_device"),
        nb::arg("trace_id"),
        R"doc(
            DRAM bytes per bank held for the trace by set_trace_capture_hold (0 when none).
        )doc");

    // Trace store (task #6183): see tt-metalium/experimental/trace_store.hpp.
    mod.def(
        "dump_trace",
        [](MeshDevice* device, MeshTraceId trace_id, const std::string& path) {
            tt::tt_metal::experimental::trace_store::dump_mesh_trace(device, trace_id, path);
        },
        nb::arg("mesh_device"),
        nb::arg("trace_id"),
        nb::arg("path"),
        nb::call_guard<nb::gil_scoped_release>(),
        R"doc(
            Write a captured trace's commands to ``path``. Fails for a trace that holds capture frees or
            captured tensor prefetcher requests.
        )doc");

    mod.def(
        "load_trace",
        [](MeshDevice* device, const std::string& path, int cq_id) {
            return tt::tt_metal::experimental::trace_store::load_mesh_trace(device, static_cast<uint8_t>(cq_id), path);
        },
        nb::arg("mesh_device"),
        nb::arg("path"),
        nb::arg("cq_id") = 0,
        nb::call_guard<nb::gil_scoped_release>(),
        R"doc(
            Install a trace written by dump_trace and write it to every device's trace region, without
            capturing it. Valid only when the same programs are loaded and every buffer the trace uses sits
            where it did at capture; the caller checks that. Returns the new trace id.
        )doc");

    mod.def(
        "trace_buffer_range",
        [](MeshDevice* device, MeshTraceId trace_id) {
            return tt::tt_metal::experimental::trace_store::mesh_trace_buffer_range(device, trace_id);
        },
        nb::arg("mesh_device"),
        nb::arg("trace_id"),
        R"doc(
            (address, size) of the trace's buffer in the trace region.
        )doc");

    mod.def(
        "allocated_buffers",
        [](MeshDevice* device) { return tt::tt_metal::experimental::trace_store::allocated_buffers(device); },
        nb::arg("mesh_device"),
        R"doc(
            Every live buffer of the mesh allocator, sorted: (buffer type, address, size, page size).
        )doc");

    mod.def(
        "set_next_allocation_address",
        [](uint64_t address, uint64_t size) {
            tt::tt_metal::experimental::trace_store::set_next_allocation_address(address, size);
        },
        nb::arg("address"),
        nb::arg("size"),
        R"doc(
            Place the next DRAM allocation of exactly ``size`` bytes at ``address`` (one-shot; other sizes
            ignore it). ``set_next_allocation_address(0, 0)`` clears it.
        )doc");

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
