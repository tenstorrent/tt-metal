// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "events.hpp"

#include <memory>
#include "tt_metal/impl/event/event.hpp"

namespace ttnn::events {

MultiDeviceEvent::MultiDeviceEvent(DeviceMesh* device_mesh) {
    TT_ASSERT(device_mesh != nullptr, "Must provide a valid device_mesh when initializing an event on multiple devices.");
    auto& devices = device_mesh->mesh_devices;
    this->events = std::vector<std::shared_ptr<Event>>(devices.size());
    for (int event_idx = 0; event_idx < devices.size(); event_idx++) {
        this->events[event_idx] = std::make_shared<Event>();
        this->events[event_idx]->device = devices[event_idx].second;
    }
}

std::shared_ptr<Event> create_event(Device* device) {
    std::shared_ptr<Event> event = std::make_shared<Event>();
    event->device = device;
    return event;
}

void record_event(uint8_t cq_id, const std::shared_ptr<Event>& event) {
    Device* device = event->device;
    device->push_work([device, event, cq_id] {
        EnqueueRecordEvent(device->command_queue(cq_id), event);
    });
}

void wait_for_event(uint8_t cq_id, const std::shared_ptr<Event>& event) {
    Device* device = event->device;
    device->push_work([device, event, cq_id] {
        EnqueueWaitForEvent(device->command_queue(cq_id), event);
    });
}

MultiDeviceEvent create_event(DeviceMesh* device_mesh) {
    return MultiDeviceEvent(device_mesh);
}

void record_event(uint8_t cq_id, const MultiDeviceEvent& multi_device_event) {
    for (auto& event : multi_device_event.events) {
        record_event(cq_id, event);
    }
}

void wait_for_event(uint8_t cq_id, const MultiDeviceEvent& multi_device_event) {
    for (auto& event : multi_device_event.events) {
        wait_for_event(cq_id, event);
    }
}


}
