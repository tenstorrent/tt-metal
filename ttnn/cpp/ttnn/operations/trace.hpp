
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/mesh_trace_id.hpp>

#include <optional>

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

}  // namespace operations::trace

}  // namespace ttnn
