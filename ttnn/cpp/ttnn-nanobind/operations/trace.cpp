// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "trace.hpp"

#include <cstdint>

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>

#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/trace.hpp"

namespace nb = nanobind;

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
        nb::overload_cast<IDevice*, QueueId>(&ttnn::operations::trace::begin_trace_capture),
        nb::arg("device"),
        nb::kw_only(),
        nb::arg("cq_id") = ttnn::DefaultQueueId);

    mod.def(
        "end_trace_capture",
        nb::overload_cast<IDevice*, uint32_t, QueueId>(&ttnn::operations::trace::end_trace_capture),
        nb::arg("device"),
        nb::arg("trace_id"),
        nb::kw_only(),
        nb::arg("cq_id") = ttnn::DefaultQueueId);

    mod.def(
        "execute_trace",
        nb::overload_cast<IDevice*, uint32_t, QueueId, bool>(&ttnn::operations::trace::execute_trace),
        nb::arg("device"),
        nb::arg("trace_id"),
        nb::kw_only(),
        nb::arg("cq_id") = ttnn::DefaultQueueId,
        nb::arg("blocking") = true);

    mod.def(
        "release_trace",
        nb::overload_cast<IDevice*, uint32_t>(&ttnn::operations::trace::release_trace),
        nb::arg("device"),
        nb::arg("trace_id"));

    // TODO: #18572 - Replace the implementation of this overload with the TT-distributed implementation.
    mod.def(
        "begin_trace_capture",
        [](MeshDevice* device, QueueId cq_id) { return ttnn::operations::trace::begin_trace_capture(device, cq_id); },
        nb::arg("mesh_device"),
        nb::kw_only(),
        nb::arg("cq_id") = ttnn::DefaultQueueId);

    // TODO: #18572 - Replace the implementation of this overload with the TT-distributed implementation.
    mod.def(
        "end_trace_capture",
        [](MeshDevice* device, uint32_t trace_id, QueueId cq_id) {
            return ttnn::operations::trace::end_trace_capture(device, trace_id, cq_id);
        },
        nb::arg("mesh_device"),
        nb::arg("trace_id"),
        nb::kw_only(),
        nb::arg("cq_id") = ttnn::DefaultQueueId);

    // TODO: #18572 - Replace the implementation of this overload with the TT-distributed implementation.
    mod.def(
        "execute_trace",
        [](MeshDevice* device, uint32_t trace_id, QueueId cq_id, bool blocking) {
            return ttnn::operations::trace::execute_trace(device, trace_id, cq_id, blocking);
        },
        nb::arg("mesh_device"),
        nb::arg("trace_id"),
        nb::kw_only(),
        nb::arg("cq_id") = ttnn::DefaultQueueId,
        nb::arg("blocking") = true);

    // TODO: #18572 - Replace the implementation of this overload with the TT-distributed implementation.
    mod.def(
        "release_trace",
        [](MeshDevice* device, uint32_t trace_id) { return ttnn::operations::trace::release_trace(device, trace_id); },
        nb::arg("mesh_device"),
        nb::arg("trace_id"));

    mod.def(
        "begin_mesh_trace_capture",
        [](MeshDevice* device, QueueId cq_id) {
            return ttnn::operations::trace::begin_mesh_trace_capture(device, cq_id);
        },
        nb::arg("mesh_device"),
        nb::kw_only(),
        nb::arg("cq_id") = ttnn::DefaultQueueId);

    mod.def(
        "end_mesh_trace_capture",
        [](MeshDevice* device, MeshTraceId trace_id, QueueId cq_id) {
            return ttnn::operations::trace::end_mesh_trace_capture(device, trace_id, cq_id);
        },
        nb::arg("mesh_device"),
        nb::arg("trace_id"),
        nb::kw_only(),
        nb::arg("cq_id") = ttnn::DefaultQueueId);

    mod.def(
        "execute_mesh_trace",
        [](MeshDevice* device, MeshTraceId trace_id, QueueId cq_id, bool blocking) {
            return ttnn::operations::trace::execute_mesh_trace(device, trace_id, cq_id, blocking);
        },
        nb::arg("mesh_device"),
        nb::arg("trace_id"),
        nb::kw_only(),
        nb::arg("cq_id") = ttnn::DefaultQueueId,
        nb::arg("blocking") = true);

    mod.def(
        "release_mesh_trace",
        [](MeshDevice* device, MeshTraceId trace_id) {
            return ttnn::operations::trace::release_mesh_trace(device, trace_id);
        },
        nb::arg("mesh_device"),
        nb::arg("trace_id"));
}

} // namespace ttnn::operations::trace
