// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/trace.hpp"

#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/mesh_trace_id.hpp>
#include <tt-metalium/distributed.hpp>

#include <tracy/Tracy.hpp>
#include "ttnn/common/queue_id.hpp"

namespace ttnn::operations::trace {

MeshTraceId begin_trace_capture(MeshDevice* device, std::optional<QueueId> cq_id) {
    ZoneScoped;
    auto cq_id_int = tt::tt_metal::raw_optional(cq_id);
    return device->begin_mesh_trace(cq_id_int);
}
void end_trace_capture(MeshDevice* device, MeshTraceId trace_id, std::optional<QueueId> cq_id) {
    ZoneScoped;
    auto cq_id_int = tt::tt_metal::raw_optional(cq_id);
    device->end_mesh_trace(cq_id_int, trace_id);
}
void execute_trace(MeshDevice* device, MeshTraceId trace_id, std::optional<QueueId> cq_id, bool blocking) {
    ZoneScoped;
    auto cq_id_int = tt::tt_metal::raw_optional(cq_id);
    device->replay_mesh_trace(cq_id_int, trace_id, blocking);
}
void release_trace(MeshDevice* device, MeshTraceId trace_id) {
    ZoneScoped;
    device->release_mesh_trace(trace_id);
}

}  // namespace ttnn::operations::trace
