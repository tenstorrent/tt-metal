
// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/mesh_trace_id.hpp>
#include <tt-metalium/distributed.hpp>

#include <optional>

#include "ttnn/types.hpp"

namespace ttnn {

using MeshTraceId = tt::tt_metal::distributed::MeshTraceId;

namespace operations::trace {

MeshTraceId begin_trace_capture(MeshDevice* device, std::optional<QueueId> cq_id);
void end_trace_capture(MeshDevice* device, MeshTraceId trace_id, std::optional<QueueId> cq_id);
void execute_trace(MeshDevice* device, MeshTraceId trace_id, std::optional<QueueId> cq_id, bool blocking);
void release_trace(MeshDevice* device, MeshTraceId trace_id);

using TraceWorkerDescData = tt::tt_metal::distributed::TraceWorkerDescExport;
using TraceExportData = tt::tt_metal::distributed::TraceExportData;

TraceExportData get_trace_data(MeshDevice* device, MeshTraceId trace_id);

std::vector<uint32_t> read_raw_buffer_data(MeshDevice* device, const Tensor& tensor);

}  // namespace operations::trace

}  // namespace ttnn
