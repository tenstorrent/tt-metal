// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/trace.hpp"

#include <tt-metalium/mesh_trace_id.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/allocation_context.hpp>
#include "tt_metal/distributed/trace_allocation_tracker.hpp"

#include <tracy/Tracy.hpp>
#include "ttnn/common/queue_id.hpp"
#include "ttnn/core.hpp"

namespace ttnn::operations::trace {

namespace tracker = tt::tt_metal::distributed::trace_allocation_tracker;

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
    tracker::mark_allocations_safe(device);
}

void mark_allocations_unsafe(MeshDevice* device, MeshTraceId trace_id) {
    ZoneScoped;
    tracker::mark_allocations_unsafe(device, trace_id);
}

bool allocations_unsafe(MeshDevice* device) {
    ZoneScoped;
    return tracker::allocations_unsafe(device);
}

std::unordered_map<size_t, std::string> get_unsafe_tracked_ids(MeshDevice* device, MeshTraceId trace_id) {
    return tracker::get_unsafe_tracked_ids(device, trace_id);
}
void remove_unsafe_tracked_id(MeshDevice* device, size_t buffer_unique_id) {
    tracker::remove_unsafe_tracked_id(device, buffer_unique_id);
}
void clear_unsafe_tracked_ids(MeshDevice* device, MeshTraceId trace_id) {
    tracker::clear_unsafe_tracked_ids(device, trace_id);
}
std::vector<size_t> drain_pending_traceback_ids() { return tracker::drain_pending_traceback_ids(); }
std::vector<size_t> drain_retired_traceback_ids() { return tracker::drain_retired_traceback_ids(); }
void push_corruptible_allocation_scope(MeshDevice* device) { tracker::push_corruptible_allocation_scope(device); }
void pop_corruptible_allocation_scope(MeshDevice* device) { tracker::pop_corruptible_allocation_scope(device); }

void push_allocation_context(const std::string& ctx) { tt::tt_metal::push_allocation_context(ctx); }
void pop_allocation_context() { tt::tt_metal::pop_allocation_context(); }

}  // namespace ttnn::operations::trace
