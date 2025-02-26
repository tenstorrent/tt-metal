// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "events.hpp"

#include <memory>
#include <tt-metalium/event.hpp>
#include "ttnn/distributed/types.hpp"

#include <tt-metalium/host_api.hpp>

using namespace tt::tt_metal;

namespace ttnn::events {

std::shared_ptr<Event> create_event(IDevice* device) {
    std::shared_ptr<Event> event = std::make_shared<Event>();
    event->device = device;
    return event;
}

void record_event(QueueId cq_id, const std::shared_ptr<Event>& event, const std::vector<SubDeviceId>& sub_device_ids) {
    IDevice* device = event->device;
    device->push_work([device, event, cq_id, sub_device_ids] {
        EnqueueRecordEvent(device->command_queue(*cq_id), event, sub_device_ids);
    });
}

void wait_for_event(QueueId cq_id, const std::shared_ptr<Event>& event) {
    IDevice* device = event->device;
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
    QueueId cq_id, const MultiDeviceEvent& multi_device_event, const std::vector<SubDeviceId>& sub_device_ids) {
    for (auto& event : multi_device_event.events) {
        record_event(cq_id, event, sub_device_ids);
    }
}

void wait_for_event(QueueId cq_id, const MultiDeviceEvent& multi_device_event) {
    for (auto& event : multi_device_event.events) {
        wait_for_event(cq_id, event);
    }
}

}  // namespace ttnn::events
