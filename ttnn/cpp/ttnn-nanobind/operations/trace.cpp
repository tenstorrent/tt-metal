// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "trace.hpp"

#include <cstdint>

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>

#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/trace.hpp"

namespace ttnn::operations::trace {

void py_module_types(nb::module_& mod) {
    nb::class_<ttnn::MeshTraceId>(mod, "MeshTraceId")
        .def(nb::init<uint32_t>())
        .def("__int__", [](const ttnn::MeshTraceId& self) { return self.get(); })
        .def(
            "__repr__", [](const ttnn::MeshTraceId& self) { return "MeshTraceId(" + std::to_string(self.get()) + ")"; })
        .def(nb::self == nb::self);
}

void py_module(nb::module_& mod) {
    mod.def(
        "begin_trace_capture",
        [](MeshDevice* device, QueueId cq_id) { return ttnn::operations::trace::begin_trace_capture(device, cq_id); },
        nb::arg("mesh_device"),
        nb::kw_only(),
        nb::arg("cq_id") = ttnn::DefaultQueueId);

    mod.def(
        "end_trace_capture",
        [](MeshDevice* device, MeshTraceId trace_id, QueueId cq_id) {
            return ttnn::operations::trace::end_trace_capture(device, trace_id, cq_id);
        },
        nb::arg("mesh_device"),
        nb::arg("trace_id"),
        nb::kw_only(),
        nb::arg("cq_id") = ttnn::DefaultQueueId);

    mod.def(
        "execute_trace",
        [](MeshDevice* device, MeshTraceId trace_id, QueueId cq_id, bool blocking) {
            return ttnn::operations::trace::execute_trace(device, trace_id, cq_id, blocking);
        },
        nb::arg("mesh_device"),
        nb::arg("trace_id"),
        nb::kw_only(),
        nb::arg("cq_id") = ttnn::DefaultQueueId,
        nb::arg("blocking") = true);

    mod.def(
        "release_trace",
        [](MeshDevice* device, MeshTraceId trace_id) {
            return ttnn::operations::trace::release_trace(device, trace_id);
        },
        nb::arg("mesh_device"),
        nb::arg("trace_id"));
}

} // namespace ttnn::operations::trace
