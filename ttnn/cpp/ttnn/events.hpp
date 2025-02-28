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

struct MultiDeviceEvent {
    MultiDeviceEvent(MeshDevice* mesh_device);
    std::vector<std::shared_ptr<tt::tt_metal::Event>> events;
};
// Single Device APIs
std::shared_ptr<tt::tt_metal::Event> create_event(tt::tt_metal::IDevice* device);
void record_event(
    QueueId cq_id,
    const std::shared_ptr<tt::tt_metal::Event>& event,
    const std::vector<tt::tt_metal::SubDeviceId>& sub_device_ids = {});
void wait_for_event(QueueId cq_id, const std::shared_ptr<tt::tt_metal::Event>& event);
// Multi Device APIs
MultiDeviceEvent create_event(MeshDevice* mesh_device);
void record_event(
    QueueId cq_id, const MultiDeviceEvent& event, const std::vector<tt::tt_metal::SubDeviceId>& sub_device_ids = {});
void wait_for_event(QueueId cq_id, const MultiDeviceEvent& event);

}  // namespace ttnn::events
