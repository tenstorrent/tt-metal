// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/trace.hpp"

#include <tt-metalium/mesh_trace_id.hpp>
#include <tt-metalium/distributed.hpp>

#include <tracy/Tracy.hpp>
#include "ttnn/common/queue_id.hpp"
#include "ttnn/core.hpp"

namespace ttnn::operations::trace {

inline bool trace_capture_disabled = false;

inline bool is_trace_capture_disabled_env_var_set() {
    const char* trace_capture_disable_str = std::getenv("TTNN_DISABLE_TRACE_CAPTURE");
    if (trace_capture_disable_str != nullptr && trace_capture_disable_str[0] == '1') {
        trace_capture_disabled = true;
    }
    return trace_capture_disabled;
}

MeshTraceId begin_trace_capture(MeshDevice* device, std::optional<QueueId> cq_id) {
    ZoneScoped;
    QueueId cq_id_value = cq_id.value_or(get_current_command_queue_id_for_thread());
    if (is_trace_capture_disabled_env_var_set()) {
        return MeshTraceId(0);
    }
    return device->begin_mesh_trace(cq_id_value.get());
}
void end_trace_capture(MeshDevice* device, MeshTraceId trace_id, std::optional<QueueId> cq_id) {
    ZoneScoped;
    QueueId cq_id_value = cq_id.value_or(get_current_command_queue_id_for_thread());
    if (is_trace_capture_disabled_env_var_set()) {
        return;
    }
    device->end_mesh_trace(cq_id_value.get(), trace_id);
}
void execute_trace(MeshDevice* device, MeshTraceId trace_id, std::optional<QueueId> cq_id, bool blocking) {
    ZoneScoped;
    QueueId cq_id_value = cq_id.value_or(get_current_command_queue_id_for_thread());
    if (is_trace_capture_disabled_env_var_set()) {
        return;
    }
    device->replay_mesh_trace(cq_id_value.get(), trace_id, blocking);
}
void release_trace(MeshDevice* device, MeshTraceId trace_id) {
    ZoneScoped;
    if (is_trace_capture_disabled_env_var_set()) {
        return;
    }
    device->release_mesh_trace(trace_id);
}

}  // namespace ttnn::operations::trace
