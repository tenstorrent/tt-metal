// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "events.hpp"

#include <memory>
#include <optional>
#include <ostream>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "ttnn/common/queue_id.hpp"
#include "ttnn/events.hpp"
#include <tt-metalium/event.hpp>

using namespace tt::tt_metal;

namespace ttnn::events {

void py_module_types(nb::module_& mod) {
    nb::class_<Event>(mod, "event");
    nb::class_<MultiDeviceEvent>(mod, "multi_device_event");
    nb::class_<MeshEvent>(mod, "MeshEvent")
        .def("__repr__", [](const MeshEvent& self) {
            std::ostringstream str;
            str << self;
            return str.str();
    });
}

void py_module(nb::module_& mod) {
    // Multi Device APIs
    mod.def(
        "record_event",
        nb::overload_cast<
            MeshDevice*,
            QueueId,
            const std::vector<SubDeviceId>&,
            const std::optional<MeshCoordinateRange>&>(&record_mesh_event),
        nb::arg("mesh_device"),
        nb::arg("cq_id"),
        nb::arg("sub_device_ids") = std::vector<SubDeviceId>(),
        nb::arg("device_range") = nb::none(),
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

    mod.def(
        "wait_for_event",
        nb::overload_cast<QueueId, const MeshEvent&>(&wait_for_mesh_event),
        nb::arg("cq_id"),
        nb::arg("mesh_event"),
        R"doc(
            Inserts a barrier - makes a CQ wait until an event is recorded.

            Args:
                cq_id (int): The Command Queue on which the barrier is being issued.
                mesh_event (MeshEvent): The Command Queue will stall until this event is completed.
            )doc");

    mod.def(
        "event_synchronize",
        nb::overload_cast<const MeshEvent&>(&event_synchronize),
        nb::arg("mesh_event"),
        R"doc(
            Synchronizes a mesh event, blocking until the event is completed.

            Args:
                mesh_event (MeshEvent): The mesh event to synchronize.
        )doc");
}

}  // namespace ttnn::events
