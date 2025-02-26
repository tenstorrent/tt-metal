// SPDX-FileCopyrightText: © 2024-2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include "ttnn/common/queue_id.hpp"
#include "ttnn/distributed/types.hpp"

#include "tt-metalium/device.hpp"
#include "tt-metalium/event.hpp"
#include "tt-metalium/sub_device_types.hpp"

namespace ttnn::events {

// Single Device APIs
std::shared_ptr<Event> create_event(IDevice* device);
void record_event(
    QueueId cq_id,
    const std::shared_ptr<Event>& event,
    const std::vector<tt::tt_metal::SubDeviceId>& sub_device_ids = {});
void wait_for_event(QueueId cq_id, const std::shared_ptr<Event>& event);

// Multi Device APIs
struct MultiDeviceEvent {
    std::vector<std::shared_ptr<Event>> events;
};
MultiDeviceEvent create_event(MeshDevice* mesh_device);
void record_event(
    QueueId cq_id, const MultiDeviceEvent& event, const std::vector<tt::tt_metal::SubDeviceId>& sub_device_ids = {});
void wait_for_event(QueueId cq_id, const MultiDeviceEvent& event);

}  // namespace ttnn::events
