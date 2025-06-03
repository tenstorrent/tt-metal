// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "trace.hpp"

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybind11/cast.h"

#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/trace.hpp"

namespace ttnn::operations::trace {

void py_module_types(py::module& module) {
    py::class_<ttnn::MeshTraceId>(module, "MeshTraceId")
        .def(py::init<uint32_t>())
        .def("__int__", [](const ttnn::MeshTraceId& self) { return static_cast<int>(*self); })
        .def(
            "__repr__",
            [](const ttnn::MeshTraceId& self) {
                return "MeshTraceId(" + std::to_string(static_cast<int>(*self)) + ")";
            })
        .def(py::self == py::self);
}

void py_module(py::module& module) {
    module.def(
        "begin_trace_capture",
        [](MeshDevice* device, QueueId cq_id) { return ttnn::operations::trace::begin_trace_capture(device, cq_id); },
        py::arg("mesh_device"),
        py::kw_only(),
        py::arg("cq_id") = ttnn::DefaultQueueId);

    module.def(
        "end_trace_capture",
        [](MeshDevice* device, MeshTraceId trace_id, QueueId cq_id) {
            return ttnn::operations::trace::end_trace_capture(device, trace_id, cq_id);
        },
        py::arg("mesh_device"),
        py::arg("trace_id"),
        py::kw_only(),
        py::arg("cq_id") = ttnn::DefaultQueueId);

    module.def(
        "execute_trace",
        [](MeshDevice* device, MeshTraceId trace_id, QueueId cq_id, bool blocking) {
            return ttnn::operations::trace::execute_trace(device, trace_id, cq_id, blocking);
        },
        py::arg("mesh_device"),
        py::arg("trace_id"),
        py::kw_only(),
        py::arg("cq_id") = ttnn::DefaultQueueId,
        py::arg("blocking") = true);

    module.def(
        "release_trace",
        [](MeshDevice* device, MeshTraceId trace_id) {
            return ttnn::operations::trace::release_trace(device, trace_id);
        },
        py::arg("mesh_device"),
        py::arg("trace_id"));
}

}  // namespace ttnn::operations::trace
