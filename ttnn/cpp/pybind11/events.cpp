// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "events.hpp"

#include <tt-metalium/event.hpp>
#include "pybind11/pybind11.h"
#include <pybind11/stl.h>

#include "ttnn/common/queue_id.hpp"

using namespace tt::tt_metal;

namespace ttnn::events {

void py_module_types(py::module& module) {
    py::class_<Event, std::shared_ptr<Event>>(module, "event");
    py::class_<MultiDeviceEvent>(module, "multi_device_event");
    py::class_<MeshEvent>(module, "MeshEvent").def("__repr__", [](const MeshEvent& self) {
        std::ostringstream str;
        str << self;
        return str.str();
    });
}

void py_module(py::module& module) {
    // Single Device APIs
    module.def(
        "record_event",
        py::overload_cast<IDevice*, QueueId, const std::vector<SubDeviceId>&>(&record_event),
        py::arg("cq_id"),
        py::arg("event"),
        py::arg("sub_device_ids") = std::vector<SubDeviceId>(),
        R"doc(
            Records the completion of commands on this CQ, preceeding this call.

            Args:
                device (ttnn.IDevice*): The device on which the event is being recorded.
                cq_id (int): The Command Queue on which event completion will be recorded.
                sub_device_ids (List[ttnn.SubDeviceId], optional): The sub-device IDs to record completion for. Defaults to sub-devices set by set_sub_device_stall_group.

            Returns:
                event: The event used to record completion of preceeding commands.
        )doc");

    module.def(
        "wait_for_event",
        py::overload_cast<QueueId, const std::shared_ptr<Event>&>(&wait_for_event),
        py::arg("cq_id"),
        py::arg("event"),
        R"doc(
            Inserts a barrier - makes a CQ wait until an event is recorded.

            Args:
                cq_id (int): The Command Queue on which the barrier is being issued.
                event (event): The Command Queue will stall until this event is completed.
            )doc");

    // Multi Device APIs
    module.def(
        "record_event",
        py::overload_cast<MeshDevice*, QueueId, const std::vector<SubDeviceId>&>(&record_event),
        py::arg("cq_id"),
        py::arg("multi_device_event"),
        py::arg("sub_device_ids") = std::vector<SubDeviceId>(),
        R"doc(
            Records the completion of commands on this CQ, preceeding this call.

            Args:
                device (ttnn.MeshDevice*): The device on which the event is being recorded.
                cq_id (int): The Command Queue on which event completion will be recorded.
                sub_device_ids (List[ttnn.SubDeviceId], optional): The sub-device IDs to record completion for. Defaults to sub-devices set by set_sub_device_stall_group.

            Returns:
                multi_device_event: The event used to record completion of preceeding commands.
        )doc");

    module.def(
        "wait_for_event",
        py::overload_cast<QueueId, const MultiDeviceEvent&>(&wait_for_event),
        py::arg("cq_id"),
        py::arg("multi_device_event"),
        R"doc(
            Inserts a barrier - makes a CQ wait until an event is recorded.

            Args:
                cq_id (int): The Command Queue on which the barrier is being issued.
                multi_device_event (multi_device_event): The Command Queue will stall until this event is completed.
            )doc");

    module.def(
        "record_mesh_event",
        py::overload_cast<
            MeshDevice*,
            QueueId,
            const std::vector<SubDeviceId>&,
            const std::optional<MeshCoordinateRange>&>(&record_mesh_event),
        py::arg("mesh_device"),
        py::arg("cq_id"),
        py::arg("sub_device_ids") = std::vector<SubDeviceId>(),
        py::arg("device_range") = std::nullopt,
        R"doc(
            Records the completion of commands on this CQ, preceeding this call.

            Args:
                mesh_device (ttnn.MeshDevice*): The device on which the event is being recorded.
                cq_id (int): The Command Queue on which event completion will be recorded.
                sub_device_ids (List[ttnn.SubDeviceId], optional): The sub-device IDs to record completion for. Defaults to sub-devices set by set_sub_device_stall_group.
                device_range (ttnn.MeshCoordinateRange, optional): The range of devices to record completion for. Defaults to all devices.

            Returns:
                MeshEvent: The event used to record completion of preceeding commands.
        )doc");

    module.def(
        "wait_for_mesh_event",
        py::overload_cast<QueueId, const MeshEvent&>(&wait_for_mesh_event),
        py::arg("cq_id"),
        py::arg("mesh_event"),
        R"doc(
            Inserts a barrier - makes a CQ wait until an event is recorded.

            Args:
                cq_id (int): The Command Queue on which the barrier is being issued.
                mesh_event (MeshEvent): The Command Queue will stall until this event is completed.
            )doc");
}

}  // namespace ttnn::events
