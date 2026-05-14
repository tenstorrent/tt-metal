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

    nb::class_<ttnn::operations::trace::TraceWorkerDescData>(mod, "TraceWorkerDescData")
        .def_ro("sub_device_id", &ttnn::operations::trace::TraceWorkerDescData::sub_device_id)
        .def_ro("num_completion_worker_cores", &ttnn::operations::trace::TraceWorkerDescData::num_completion_worker_cores)
        .def_ro("num_mcast_programs", &ttnn::operations::trace::TraceWorkerDescData::num_mcast_programs)
        .def_ro("num_unicast_programs", &ttnn::operations::trace::TraceWorkerDescData::num_unicast_programs);

    nb::class_<ttnn::operations::trace::TraceExportData>(mod, "TraceExportData")
        .def_ro("worker_descs", &ttnn::operations::trace::TraceExportData::worker_descs)
        .def_ro("trace_streams", &ttnn::operations::trace::TraceExportData::trace_streams)
        .def_ro("trace_buf_address", &ttnn::operations::trace::TraceExportData::trace_buf_address)
        .def_ro("trace_buf_page_size", &ttnn::operations::trace::TraceExportData::trace_buf_page_size)
        .def_ro("trace_buf_num_pages", &ttnn::operations::trace::TraceExportData::trace_buf_num_pages);
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
        "get_trace_data",
        [](MeshDevice* device, MeshTraceId trace_id) {
            return ttnn::operations::trace::get_trace_data(device, trace_id);
        },
        nb::arg("mesh_device"),
        nb::arg("trace_id"),
        "Get internal trace data for serialization. Returns TraceExportData containing "
        "worker descriptors, raw dispatch command streams, and trace buffer placement.");

    mod.def(
        "read_raw_buffer_data",
        [](MeshDevice* device, ttnn::Tensor& tensor) {
            return ttnn::operations::trace::read_raw_buffer_data(device, tensor);
        },
        nb::arg("mesh_device"),
        nb::arg("tensor"),
        "Read raw device buffer data as uint32 vector. Returns the exact DRAM/L1 bytes "
        "without detilization or dtype conversion.");
}

} // namespace ttnn::operations::trace
