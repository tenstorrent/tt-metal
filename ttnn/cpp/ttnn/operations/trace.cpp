// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/trace.hpp"

#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/mesh_trace_id.hpp>
#include <tt-metalium/distributed.hpp>

#include <tracy/Tracy.hpp>
#include "ttnn/common/queue_id.hpp"
#include "ttnn/core.hpp"

namespace ttnn::operations::trace {

MeshTraceId begin_trace_capture(MeshDevice* device, std::optional<QueueId> cq_id) {
    ZoneScoped;
    QueueId cq_id_value = cq_id.value_or(get_current_command_queue_id_for_thread());
    return device->begin_mesh_trace(cq_id_value.get());
}
void end_trace_capture(MeshDevice* device, MeshTraceId trace_id, std::optional<QueueId> cq_id) {
    ZoneScoped;
    QueueId cq_id_value = cq_id.value_or(get_current_command_queue_id_for_thread());
    device->end_mesh_trace(cq_id_value.get(), trace_id);
}
void execute_trace(MeshDevice* device, MeshTraceId trace_id, std::optional<QueueId> cq_id, bool blocking) {
    ZoneScoped;
    QueueId cq_id_value = cq_id.value_or(get_current_command_queue_id_for_thread());
    device->replay_mesh_trace(cq_id_value.get(), trace_id, blocking);
}
void release_trace(MeshDevice* device, MeshTraceId trace_id) {
    ZoneScoped;
    device->release_mesh_trace(trace_id);
}

}  // namespace ttnn::operations::trace
