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

using ::tt::tt_metal::distributed::EventSynchronize;

MeshEvent record_mesh_event(
    MeshDevice* mesh_device,
    QueueId cq_id,
    const std::vector<tt::tt_metal::SubDeviceId>& sub_device_ids,
    const std::optional<ttnn::MeshCoordinateRange>& device_range) {
    return mesh_device->mesh_command_queue(*cq_id).enqueue_record_event_to_host(sub_device_ids, device_range);
}

void wait_for_mesh_event(QueueId cq_id, const MeshEvent& event) {
    event.device()->mesh_command_queue(*cq_id).enqueue_wait_for_event(event);
}

void event_synchronize(const MeshEvent& event) { EventSynchronize(event); }

}  // namespace ttnn::events
