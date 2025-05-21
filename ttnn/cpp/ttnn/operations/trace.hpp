
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

MeshTraceId begin_trace_capture(MeshDevice* device, QueueId cq_id);
void end_trace_capture(MeshDevice* device, MeshTraceId trace_id, QueueId cq_id);
void execute_trace(MeshDevice* device, MeshTraceId trace_id, QueueId cq_id, bool blocking);
void release_trace(MeshDevice* device, MeshTraceId trace_id);

}  // namespace trace
}  // namespace operations
}  // namespace ttnn
