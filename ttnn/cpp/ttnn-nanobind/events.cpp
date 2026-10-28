// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "events.hpp"

#include <optional>
#include <sstream>
#include <vector>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/experimental/queue_fence.hpp>

#include "ttnn/common/queue_id.hpp"
#include "ttnn/events.hpp"

using namespace tt::tt_metal;

// NOLINTBEGIN(bugprone-unused-raii)

namespace ttnn::events {

void py_module_types(nb::module_& mod) {
    nb::class_<MeshEvent>(mod, "MeshEvent")
        .def("__repr__", [](const MeshEvent& self) {
            std::ostringstream str;
            str << self;
            return str.str();
    });
    nb::class_<experimental::QueueFence>(mod, "QueueFence")
        .def_ro("cq_id", &experimental::QueueFence::cq_id)
        .def_ro("count", &experimental::QueueFence::count)
        .def("__repr__", [](const experimental::QueueFence& self) {
            return fmt::format(
                "QueueFence(cq_id={}, sub_device_id={}, epoch={}, count={})",
                self.cq_id,
                self.sub_device_id.has_value() ? std::to_string(**self.sub_device_id) : "transfers",
                self.epoch,
                self.count);
        });
}

void py_module(nb::module_& mod) {
    // Multi Device APIs
    mod.def(
        "record_event",
        nb::overload_cast<
            MeshDevice*,
            std::optional<QueueId>,
            const std::vector<SubDeviceId>&,
            const std::optional<MeshCoordinateRange>&>(&record_mesh_event),
        nb::arg("mesh_device"),
        nb::arg("cq_id") = nb::none(),
        nb::arg("sub_device_ids") = std::vector<SubDeviceId>(),
        nb::arg("device_range") = nb::none(),
        R"doc(
            Records the completion of commands on this CQ, preceding this call.

            Args:
                mesh_device (ttnn.MeshDevice*): The device on which the event is being recorded.
                cq_id (int): The Command Queue on which event completion will be recorded.
                sub_device_ids (List[ttnn.SubDeviceId], optional): The sub-device IDs to record completion for. Defaults to sub-devices set by set_sub_device_stall_group.
                device_range (ttnn.MeshCoordinateRange, optional): The range of devices to record completion for. Defaults to all devices.

            Returns:
                MeshEvent: The event used to record completion of preceding commands.
        )doc");

    mod.def(
        "wait_for_event",
        nb::overload_cast<std::optional<QueueId>, const MeshEvent&>(&wait_for_mesh_event),
        nb::arg("cq_id") = nb::none(),
        nb::arg("mesh_event"),
        R"doc(
            Inserts a barrier - makes a CQ wait until an event is recorded.

            Args:
                cq_id (int): The Command Queue on which the barrier is being issued.
                mesh_event (MeshEvent): The Command Queue will stall until this event is completed.
            )doc");

    // Experimental: precise fences (tt-metalium/experimental/queue_fence.hpp).
    mod.def(
        "queue_fence",
        [](MeshDevice* mesh_device, std::optional<QueueId> cq_id, SubDeviceId sub_device_id) {
            return experimental::GetQueueFence(mesh_device->mesh_command_queue(raw_optional(cq_id)), sub_device_id);
        },
        nb::arg("mesh_device"),
        nb::arg("cq_id") = nb::none(),
        nb::arg("sub_device_id") = SubDeviceId{0},
        "Experimental: the fence after everything the queue has enqueued so far on the sub-device. Enqueues nothing.");
    mod.def(
        "wait_for_queue_fence",
        [](MeshDevice* mesh_device, std::optional<QueueId> cq_id, const experimental::QueueFence& fence) {
            experimental::EnqueueWaitForQueueFence(mesh_device->mesh_command_queue(raw_optional(cq_id)), fence);
        },
        nb::arg("mesh_device"),
        nb::arg("cq_id") = nb::none(),
        nb::arg("fence"),
        "Experimental: the queue runs nothing enqueued after this call until the fence completes.");

    mod.def(
        "event_synchronize",
        nb::overload_cast<const MeshEvent&>(&event_synchronize),
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("mesh_event"),
        R"doc(
            Synchronizes a mesh event, blocking until the event is completed.

            Args:
                mesh_event (MeshEvent): The mesh event to synchronize.
        )doc");
}

}  // namespace ttnn::events

// NOLINTEND(bugprone-unused-raii)
