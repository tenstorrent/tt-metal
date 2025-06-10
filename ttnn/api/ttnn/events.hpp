// SPDX-FileCopyrightText: © 2024-2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include "tt-metalium/mesh_event.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/distributed/types.hpp"

#include "tt-metalium/device.hpp"
#include "tt-metalium/event.hpp"
#include "tt-metalium/sub_device_types.hpp"

namespace ttnn {

using MeshEvent = tt::tt_metal::distributed::MeshEvent;

namespace events {

// Single Device APIs
std::shared_ptr<tt::tt_metal::Event> record_event(
    tt::tt_metal::IDevice* device, QueueId cq_id, const std::vector<tt::tt_metal::SubDeviceId>& sub_device_ids = {});
void wait_for_event(QueueId cq_id, const std::shared_ptr<tt::tt_metal::Event>& event);

// Multi Device APIs
struct MultiDeviceEvent {
    std::vector<std::shared_ptr<tt::tt_metal::Event>> events;
};
MultiDeviceEvent record_event(
    MeshDevice* mesh_device, QueueId cq_id, const std::vector<tt::tt_metal::SubDeviceId>& sub_device_ids = {});
void wait_for_event(QueueId cq_id, const MultiDeviceEvent& event);

MeshEvent record_mesh_event(
    MeshDevice* mesh_device,
    QueueId cq_id,
    const std::vector<tt::tt_metal::SubDeviceId>& sub_device_ids = {},
    const std::optional<ttnn::MeshCoordinateRange>& device_range = std::nullopt);
void wait_for_mesh_event(QueueId cq_id, const MeshEvent& event);

void event_synchronize(const MeshEvent& event);

}  // namespace events
}  // namespace ttnn
