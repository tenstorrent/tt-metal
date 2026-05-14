// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/trace.hpp"

#include <tt-metalium/mesh_trace_id.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_buffer.hpp>

#include <tracy/Tracy.hpp>
#include "ttnn/common/queue_id.hpp"
#include "ttnn/core.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/storage.hpp"

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

TraceExportData get_trace_data(MeshDevice* device, MeshTraceId trace_id) {
    ZoneScoped;
    return tt::tt_metal::distributed::GetTraceExportData(device, trace_id);
}

std::vector<uint32_t> read_raw_buffer_data(MeshDevice* device, const Tensor& tensor) {
    ZoneScoped;
    auto mesh_buffer = tensor.device_storage().get_mesh_buffer_leak_ownership();
    TT_FATAL(mesh_buffer != nullptr, "Tensor does not have a device buffer");
    auto& cq = device->mesh_command_queue();
    uint32_t buf_size_bytes = mesh_buffer->page_size() * mesh_buffer->num_pages();
    std::vector<uint32_t> result(buf_size_bytes / sizeof(uint32_t), 0);
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq, result, mesh_buffer, true);
    return result;
}

}  // namespace ttnn::operations::trace
