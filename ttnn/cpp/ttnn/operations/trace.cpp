// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/trace.hpp"

#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/mesh_trace_id.hpp>
#include <tt-metalium/distributed.hpp>

#include <tracy/Tracy.hpp>

namespace ttnn::operations::trace {

MeshTraceId begin_trace_capture(MeshDevice* device, QueueId cq_id) {
    ZoneScoped;
    return BeginTraceCapture(device, *cq_id);
}
void end_trace_capture(MeshDevice* device, MeshTraceId trace_id, QueueId cq_id) {
    ZoneScoped;
    device->end_mesh_trace(*cq_id, trace_id);
}
void execute_trace(MeshDevice* device, MeshTraceId trace_id, QueueId cq_id, bool blocking) {
    ZoneScoped;
    device->replay_mesh_trace(*cq_id, trace_id, blocking);
}
void release_trace(MeshDevice* device, MeshTraceId trace_id) {
    ZoneScoped;
    device->release_mesh_trace(trace_id);
}

}  // namespace ttnn::operations::trace
