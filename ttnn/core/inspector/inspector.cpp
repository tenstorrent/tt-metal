// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/inspector/inspector.hpp"
#include "data.hpp"
#include "tt-metalium/experimental/inspector.hpp"
#include <mutex>

namespace ttnn::inspector {

bool is_enabled() { return tt::tt_metal::experimental::inspector::IsEnabled(); }

void register_inspector_rpc() {
    if (!is_enabled()) {
        return;
    }

    static std::once_flag register_flag;

    std::call_once(register_flag, []() { Data::instance().register_inspector_rpc(); });
}

void EmitMeshWorkloadAnnotation(
    tt::tt_metal::distributed::MeshWorkload& mesh_workload,
    std::string_view operation_name,
    std::string_view operation_parameters) {
    if (!is_enabled()) {
        return;
    }
    try {
        auto& data = Data::instance();
        std::lock_guard<std::mutex> lock(data.mesh_workloads_mutex);
        auto& mesh_workload_data = data.mesh_workloads_data[mesh_workload.get_id()];
        mesh_workload_data.name = std::string(operation_name);
        mesh_workload_data.parameters = std::string(operation_parameters);
    } catch (const std::exception& e) {
        // TODO: TT_INSPECTOR_LOG("Failed to log mesh workload set metadata: {}", e.what());
    }
}

void EmitMeshWorkloadRuntimeId(tt::tt_metal::distributed::MeshWorkload& mesh_workload, uint64_t runtime_id) {
    if (!is_enabled()) {
        return;
    }
    try {
        auto& data = Data::instance();

        std::lock_guard<std::mutex> lock(data.runtime_ids_mutex);
        data.runtime_ids.push_back({mesh_workload.get_id(), runtime_id});

        // Keep only the last MAX_RUNTIME_ID_ENTRIES
        if (data.runtime_ids.size() > inspector::Data::MAX_RUNTIME_ID_ENTRIES) {
            data.runtime_ids.pop_front();
        }
    } catch (const std::exception& e) {
        // TODO: TT_INSPECTOR_LOG("Failed to log workload runtime ID: {}", e.what());
    }
}

}  // namespace ttnn::inspector
