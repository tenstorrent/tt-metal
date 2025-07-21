// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "events.hpp"

#include <memory>
#include <tt-metalium/event.hpp>
#include "tt-metalium/distributed.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/distributed/types.hpp"

#include <tt-metalium/host_api.hpp>

namespace ttnn::events {

using ::tt::tt_metal::EnqueueRecordEvent;
using ::tt::tt_metal::EnqueueWaitForEvent;
using ::tt::tt_metal::distributed::EnqueueRecordEventToHost;
using ::tt::tt_metal::distributed::EnqueueWaitForEvent;
using ::tt::tt_metal::distributed::EventSynchronize;

std::shared_ptr<tt::tt_metal::Event> record_event(
    tt::tt_metal::IDevice* device, QueueId cq_id, const std::vector<tt::tt_metal::SubDeviceId>& sub_device_ids) {
    std::shared_ptr<tt::tt_metal::Event> event = std::make_shared<tt::tt_metal::Event>();
    EnqueueRecordEvent(device->command_queue(*cq_id), event, sub_device_ids);
    return event;
}

void wait_for_event(QueueId cq_id, const std::shared_ptr<tt::tt_metal::Event>& event) {
    tt::tt_metal::IDevice* device = event->device;
    EnqueueWaitForEvent(device->command_queue(*cq_id), event);
}

MultiDeviceEvent record_event(
    MeshDevice* mesh_device, QueueId cq_id, const std::vector<tt::tt_metal::SubDeviceId>& sub_device_ids) {
    MultiDeviceEvent multi_device_event;
    multi_device_event.events.reserve(mesh_device->get_devices().size());
    for (auto* device : mesh_device->get_devices()) {
        multi_device_event.events.push_back(record_event(device, cq_id, sub_device_ids));
    }
    return multi_device_event;
}

void wait_for_event(QueueId cq_id, const MultiDeviceEvent& multi_device_event) {
    for (auto& event : multi_device_event.events) {
        wait_for_event(cq_id, event);
    }
}

MeshEvent record_mesh_event(
    MeshDevice* mesh_device,
    QueueId cq_id,
    const std::vector<tt::tt_metal::SubDeviceId>& sub_device_ids,
    const std::optional<ttnn::MeshCoordinateRange>& device_range) {
    return EnqueueRecordEventToHost(mesh_device->mesh_command_queue(*cq_id), sub_device_ids, device_range);
}

void wait_for_mesh_event(QueueId cq_id, const MeshEvent& event) {
    EnqueueWaitForEvent(event.device()->mesh_command_queue(*cq_id), event);
}

void event_synchronize(const MeshEvent& event) { EventSynchronize(event); }

}  // namespace ttnn::events
