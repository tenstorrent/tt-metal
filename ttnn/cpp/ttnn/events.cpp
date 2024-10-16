// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "events.hpp"

#include <memory>
#include "tt_metal/impl/event/event.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::events {

MultiDeviceEvent::MultiDeviceEvent(MeshDevice* mesh_device) {
    TT_ASSERT(mesh_device != nullptr,
              "Must provide a valid mesh_device when initializing an event on multiple devices.");
    auto devices = mesh_device->get_devices();
    this->events = std::vector<std::shared_ptr<Event>>(devices.size());
    for (int event_idx = 0; event_idx < devices.size(); event_idx++) {
        this->events[event_idx] = std::make_shared<Event>();
        this->events[event_idx]->device = devices[event_idx];
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

MultiDeviceEvent create_event(MeshDevice* mesh_device) {
    return MultiDeviceEvent(mesh_device);
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

}  // namespace ttnn::events
