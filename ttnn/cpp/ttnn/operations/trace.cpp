// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/trace.hpp"

#include <tt-metalium/mesh_trace_id.hpp>
#include <tt-metalium/distributed.hpp>

#include <tracy/Tracy.hpp>
#include "ttnn/common/queue_id.hpp"
#include "ttnn/core.hpp"

namespace ttnn::operations::trace {

// A missing cq_id resolves to the thread's current command queue (ttnn::core::current_mesh_command_queue).
MeshTraceId begin_trace_capture(MeshDevice* device, std::optional<QueueId> cq_id) {
    ZoneScoped;
    return device->begin_mesh_trace(ttnn::core::current_mesh_command_queue(*device, cq_id));
}
void end_trace_capture(MeshDevice* device, MeshTraceId trace_id, std::optional<QueueId> cq_id) {
    ZoneScoped;
    device->end_mesh_trace(ttnn::core::current_mesh_command_queue(*device, cq_id), trace_id);
}
void execute_trace(MeshDevice* device, MeshTraceId trace_id, std::optional<QueueId> cq_id, bool blocking) {
    ZoneScoped;
    device->replay_mesh_trace(ttnn::core::current_mesh_command_queue(*device, cq_id), trace_id, blocking);
}
void release_trace(MeshDevice* device, MeshTraceId trace_id) {
    ZoneScoped;
    device->release_mesh_trace(trace_id);
}

}  // namespace ttnn::operations::trace
