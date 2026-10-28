// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "events.hpp"

#include <memory>
#include "tt-metalium/distributed.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/queue_hazards.hpp"

namespace ttnn::events {

using ::tt::tt_metal::distributed::EventSynchronize;

MeshEvent record_mesh_event(
    MeshDevice* mesh_device,
    std::optional<QueueId> cq_id,
    const std::vector<tt::tt_metal::SubDeviceId>& sub_device_ids,
    const std::optional<ttnn::MeshCoordinateRange>& device_range) {
    auto event =
        mesh_device->mesh_command_queue(raw_optional(cq_id)).enqueue_record_event_to_host(sub_device_ids, device_range);
    if (queue_hazards::enabled()) {
        queue_hazards::on_record_event(event, sub_device_ids);
    }
    return event;
}

void wait_for_mesh_event(std::optional<QueueId> cq_id, const MeshEvent& event) {
    auto& queue = event.device()->mesh_command_queue(raw_optional(cq_id));
    queue.enqueue_wait_for_event(event);
    if (queue_hazards::enabled()) {
        queue_hazards::on_wait_for_event(QueueId(queue.id()), event);
    }
}

void event_synchronize(const MeshEvent& event) {
    EventSynchronize(event);
    if (queue_hazards::enabled()) {
        queue_hazards::on_event_synchronize(event);
    }
}

}  // namespace ttnn::events
