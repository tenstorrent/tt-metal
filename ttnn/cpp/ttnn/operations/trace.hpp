
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/mesh_trace_id.hpp>

#include "ttnn/types.hpp"

namespace ttnn {

using MeshTraceId = tt::tt_metal::distributed::MeshTraceId;

namespace operations {
namespace trace {

// Trace APIs - Single Device
uint32_t begin_trace_capture(IDevice* device, QueueId cq_id);
void end_trace_capture(IDevice* device, uint32_t trace_id, QueueId cq_id);
void execute_trace(IDevice* device, uint32_t trace_id, QueueId cq_id, bool blocking);
void release_trace(IDevice* device, uint32_t trace_id);

// Trace APIs - Multi Device
MeshTraceId begin_mesh_trace_capture(MeshDevice* device, QueueId cq_id);
void end_mesh_trace_capture(MeshDevice* device, MeshTraceId trace_id, QueueId cq_id);
void execute_mesh_trace(MeshDevice* device, MeshTraceId trace_id, QueueId cq_id, bool blocking);
void release_mesh_trace(MeshDevice* device, MeshTraceId trace_id);

}  // namespace trace
}  // namespace operations
}  // namespace ttnn
