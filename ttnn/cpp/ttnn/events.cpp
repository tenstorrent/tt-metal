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

std::shared_ptr<tt::tt_metal::Event> create_event(tt::tt_metal::IDevice* device) {
    std::shared_ptr<tt::tt_metal::Event> event = std::make_shared<tt::tt_metal::Event>();
    event->device = device;
    return event;
}

void record_event(
    QueueId cq_id,
    const std::shared_ptr<tt::tt_metal::Event>& event,
    const std::vector<tt::tt_metal::SubDeviceId>& sub_device_ids) {
    tt::tt_metal::IDevice* device = event->device;
    device->push_work([device, event, cq_id, sub_device_ids] {
        EnqueueRecordEvent(device->command_queue(*cq_id), event, sub_device_ids);
    });
}

void wait_for_event(QueueId cq_id, const std::shared_ptr<tt::tt_metal::Event>& event) {
    tt::tt_metal::IDevice* device = event->device;
    device->push_work([device, event, cq_id] { EnqueueWaitForEvent(device->command_queue(*cq_id), event); });
}

MultiDeviceEvent create_event(MeshDevice* mesh_device) {
    MultiDeviceEvent multi_device_event;

    multi_device_event.events.reserve(mesh_device->get_devices().size());
    for (auto* device : mesh_device->get_devices()) {
        multi_device_event.events.push_back(create_event(device));
    }
    return multi_device_event;
}

void record_event(
    QueueId cq_id,
    const MultiDeviceEvent& multi_device_event,
    const std::vector<tt::tt_metal::SubDeviceId>& sub_device_ids) {
    for (auto& event : multi_device_event.events) {
        record_event(cq_id, event, sub_device_ids);
    }
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

}  // namespace ttnn::events
