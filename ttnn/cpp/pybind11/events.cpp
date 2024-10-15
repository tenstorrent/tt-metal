// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "events.hpp"

#include "tt_metal/impl/event/event.hpp"
#include "pybind11/pybind11.h"

namespace ttnn::events {

void py_module_types(py::module& module) {
    py::class_<Event, std::shared_ptr<Event>>(module, "event");
    py::class_<MultiDeviceEvent>(module, "multi_device_event");
}

void py_module(py::module& module) {
    // Single Device APIs
    module.def(
        "create_event",
        py::overload_cast<Device*>(&create_event),
        py::arg("device"),
        R"doc(
            Create an Event Object on a single device.

            Args:
                device (Device): The device on which this event will be used for synchronization.
            )doc");

    module.def(
        "record_event",
        py::overload_cast<uint8_t, const std::shared_ptr<Event>&>(&record_event),
        py::arg("cq_id"),
        py::arg("event"),
        R"doc(
            Record the completion of commands on this CQ, preceeding this call.

            Args:
                cq_id (int): The Command Queue on which event completion will be recorded.
                event (event): The event used to record completion of preceeding commands.
            )doc");

    module.def(
        "wait_for_event",
        py::overload_cast<uint8_t, const std::shared_ptr<Event>&>(&wait_for_event),
        py::arg("cq_id"),
        py::arg("event"),
        R"doc(
            Insert a barrier - Make a CQ wait until an event is recorded.

            Args:
                cq_id (int): The Command Queue on which the barrier is being issued.
                event (event): The Command Queue will stall until this event is completed.
            )doc");

    // Multi Device APIs
    module.def(
        "create_event",
        py::overload_cast<MeshDevice*>(&create_event),
        py::arg("mesh_device"),
        R"doc(
            Create an Event Object on a mesh of devices.

            Args:
                mesh_device (MeshDevice): The mesh on which this event will be used for synchronization.
            )doc");

    module.def(
        "record_event",
        py::overload_cast<uint8_t, const MultiDeviceEvent&>(&record_event),
        py::arg("cq_id"),
        py::arg("multi_device_event"),
        R"doc(
            Record the completion of commands on this CQ, preceeding this call.

            Args:
                cq_id (int): The Command Queue on which event completion will be recorded.
                event (event): The event used to record completion of preceeding commands.
            )doc");

    module.def(
        "wait_for_event",
        py::overload_cast<uint8_t, const MultiDeviceEvent&>(&wait_for_event),
        py::arg("cq_id"),
        py::arg("multi_device_event"),
        R"doc(
            Record the completion of commands on this CQ, preceeding this call.

            Args:
                cq_id (int): The Command Queue on which event completion will be recorded.
                event (event): The event used to record completion of preceeding commands.
            )doc");
}

}  // namespace ttnn::events
