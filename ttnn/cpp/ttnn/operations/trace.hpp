
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/mesh_trace_id.hpp>

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "ttnn/types.hpp"

namespace ttnn {

using MeshTraceId = tt::tt_metal::distributed::MeshTraceId;

namespace operations::trace {

MeshTraceId begin_trace_capture(MeshDevice* device, std::optional<QueueId> cq_id);
void end_trace_capture(MeshDevice* device, MeshTraceId trace_id, std::optional<QueueId> cq_id);
void execute_trace(MeshDevice* device, MeshTraceId trace_id, std::optional<QueueId> cq_id, bool blocking);
void release_trace(MeshDevice* device, MeshTraceId trace_id);
void mark_allocations_safe(MeshDevice* device);
void mark_allocations_unsafe(MeshDevice* device);
bool allocations_unsafe(MeshDevice* device);

// Unsafe allocation tracking
std::unordered_map<size_t, std::string> get_unsafe_tracked_ids(MeshDevice* device);
void clear_unsafe_tracked_ids(MeshDevice* device);
std::vector<size_t> drain_pending_traceback_ids();

// Thread-local allocation context stack (delegates to tt::tt_metal:: free functions)
void push_allocation_context(const std::string& ctx);
void pop_allocation_context();

}  // namespace operations::trace

}  // namespace ttnn
