// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "trace.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/unordered_set.h>
#include <nanobind/stl/vector.h>

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
}

void py_module(nb::module_& mod) {
    mod.def(
        "begin_trace_capture",
        [](MeshDevice* device, std::optional<ttnn::QueueId> cq_id) {
            return ttnn::operations::trace::begin_trace_capture(device, cq_id);
        },
        nb::arg("mesh_device"),
        nb::kw_only(),
        nb::arg("cq_id") = nb::none());

    mod.def(
        "end_trace_capture",
        [](MeshDevice* device, MeshTraceId trace_id, std::optional<ttnn::QueueId> cq_id) {
            ttnn::operations::trace::end_trace_capture(device, trace_id, cq_id);
        },
        nb::arg("mesh_device"),
        nb::arg("trace_id"),
        nb::kw_only(),
        nb::arg("cq_id") = nb::none());

    mod.def(
        "execute_trace",
        [](MeshDevice* device, MeshTraceId trace_id, std::optional<QueueId> cq_id, bool blocking) {
            ttnn::operations::trace::execute_trace(device, trace_id, cq_id, blocking);
        },
        nb::arg("mesh_device"),
        nb::arg("trace_id"),
        nb::kw_only(),
        nb::arg("cq_id") = nb::none(),
        nb::arg("blocking") = true);

    mod.def(
        "release_trace",
        [](MeshDevice* device, MeshTraceId trace_id) { ttnn::operations::trace::release_trace(device, trace_id); },
        nb::arg("mesh_device"),
        nb::arg("trace_id"));

    mod.def(
        "mark_allocations_safe",
        [](MeshDevice* device) { ttnn::operations::trace::mark_allocations_safe(device); },
        nb::arg("mesh_device"));

    mod.def(
        "mark_allocations_unsafe",
        [](MeshDevice* device) { ttnn::operations::trace::mark_allocations_unsafe(device); },
        nb::arg("mesh_device"));

    mod.def(
        "allocations_unsafe",
        [](MeshDevice* device) { return ttnn::operations::trace::allocations_unsafe(device); },
        nb::arg("mesh_device"));

    // Unsafe allocation tracking
    mod.def(
        "suppress_unsafe_allocation_warning",
        [](MeshDevice* device) { ttnn::operations::trace::suppress_unsafe_allocation_warning(device); },
        nb::arg("mesh_device"));
    mod.def(
        "unsuppress_unsafe_allocation_warning",
        [](MeshDevice* device) { ttnn::operations::trace::unsuppress_unsafe_allocation_warning(device); },
        nb::arg("mesh_device"));
    mod.def(
        "get_unsafe_tracked_ids",
        [](MeshDevice* device) { return ttnn::operations::trace::get_unsafe_tracked_ids(device); },
        nb::arg("mesh_device"));
    mod.def(
        "clear_unsafe_tracked_ids",
        [](MeshDevice* device) { ttnn::operations::trace::clear_unsafe_tracked_ids(device); },
        nb::arg("mesh_device"));
    mod.def("drain_pending_traceback_ids", []() { return ttnn::operations::trace::drain_pending_traceback_ids(); });
}

} // namespace ttnn::operations::trace
