// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/trace.hpp"

#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/trace.hpp>
#include <tt-metalium/mesh_trace_id.hpp>
#include <tt-metalium/mesh_trace.hpp>

namespace ttnn::operations::trace {

uint32_t begin_trace_capture(IDevice* device, QueueId cq_id) {
    ZoneScoped;
    uint32_t trace_id = tt::tt_metal::Trace::next_id();
    device->begin_trace(*cq_id, trace_id);
    return trace_id;
}

void end_trace_capture(IDevice* device, uint32_t trace_id, QueueId cq_id) {
    ZoneScoped;
    device->end_trace(*cq_id, trace_id);
}

void execute_trace(IDevice* device, uint32_t trace_id, QueueId cq_id, bool blocking) {
    ZoneScoped;
    device->replay_trace(*cq_id, trace_id, blocking /* block_on_device */, blocking /* block_on_worker_thread */);
}

void release_trace(IDevice* device, uint32_t trace_id) {
    ZoneScoped;
    device->release_trace(trace_id);
}

MeshTraceId begin_mesh_trace_capture(MeshDevice* device, QueueId cq_id) {
    ZoneScoped;
    MeshTraceId trace_id = tt::tt_metal::distributed::MeshTrace::next_id();
    device->begin_mesh_trace(*cq_id, trace_id);
    return trace_id;
}
void end_mesh_trace_capture(MeshDevice* device, MeshTraceId trace_id, QueueId cq_id) {
    ZoneScoped;
    device->end_mesh_trace(*cq_id, trace_id);
}
void execute_mesh_trace(MeshDevice* device, MeshTraceId trace_id, QueueId cq_id, bool blocking) {
    ZoneScoped;
    device->replay_mesh_trace(*cq_id, trace_id, blocking);
}
void release_mesh_trace(MeshDevice* device, MeshTraceId trace_id) {
    ZoneScoped;
    device->release_mesh_trace(trace_id);
}

}  // namespace ttnn::operations::trace
