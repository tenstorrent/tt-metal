// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/trace.hpp"

#include <tt-metalium/mesh_trace_id.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/allocation_context.hpp>

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

void mark_allocations_safe(MeshDevice* device) {
    ZoneScoped;
    device->mark_allocations_safe();
}

void mark_allocations_unsafe(MeshDevice* device) {
    ZoneScoped;
    device->mark_allocations_unsafe();
}

bool allocations_unsafe(MeshDevice* device) {
    ZoneScoped;
    return device->allocations_unsafe();
}

std::unordered_map<size_t, std::string> get_unsafe_tracked_ids(MeshDevice* device) {
    return device->get_unsafe_tracked_ids();
}
void clear_unsafe_tracked_ids(MeshDevice* device) { device->clear_unsafe_tracked_ids(); }
std::vector<size_t> drain_pending_traceback_ids() { return MeshDevice::drain_pending_traceback_ids(); }

void push_allocation_context(const std::string& ctx) { tt::tt_metal::push_allocation_context(ctx); }
void pop_allocation_context() { tt::tt_metal::pop_allocation_context(); }

}  // namespace ttnn::operations::trace
