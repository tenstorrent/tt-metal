// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "tt_metal/impl/device/device_mesh.hpp"

namespace ttnn::events {

struct MultiDeviceEvent
{
    MultiDeviceEvent(DeviceMesh* device_mesh);
    std::vector<std::shared_ptr<Event>> events;
};
// Single Device APIs
std::shared_ptr<Event> create_event(Device* device);
void record_event(uint8_t cq_id, const std::shared_ptr<Event>& event);
void wait_for_event(uint8_t cq_id, const std::shared_ptr<Event>& event);
// Multi Device APIs
MultiDeviceEvent create_event(DeviceMesh* device_mesh);
void record_event(uint8_t cq_id, const MultiDeviceEvent& event);
void wait_for_event(uint8_t cq_id, const MultiDeviceEvent& event);

} // namespace ttnn::events
