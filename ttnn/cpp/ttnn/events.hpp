// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include "tt_metal/host_api.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::events {

struct MultiDeviceEvent
{
    MultiDeviceEvent(MeshDevice* mesh_device);
    std::vector<std::shared_ptr<Event>> events;
};
// Single Device APIs
std::shared_ptr<Event> create_event(Device* device);
void record_event(uint8_t cq_id, const std::shared_ptr<Event>& event);
void wait_for_event(uint8_t cq_id, const std::shared_ptr<Event>& event);
// Multi Device APIs
MultiDeviceEvent create_event(MeshDevice* mesh_device);
void record_event(uint8_t cq_id, const MultiDeviceEvent& event);
void wait_for_event(uint8_t cq_id, const MultiDeviceEvent& event);

} // namespace ttnn::events
