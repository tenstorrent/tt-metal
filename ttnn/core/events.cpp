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

using ::tt::tt_metal::EnqueueWaitForEvent;
using ::tt::tt_metal::distributed::EnqueueRecordEventToHost;
using ::tt::tt_metal::distributed::EnqueueWaitForEvent;
using ::tt::tt_metal::distributed::EventSynchronize;

void wait_for_event(QueueId cq_id, const std::shared_ptr<tt::tt_metal::Event>& event) {
    tt::tt_metal::IDevice* device = event->device;
    EnqueueWaitForEvent(device->command_queue(*cq_id), event);
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
