// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "data.hpp"

#include "tt-metalium/experimental/inspector.hpp"

namespace ttnn::inspector {

Data& Data::instance() {
    static Data instance;
    return instance;
}

void Data::register_inspector_rpc() {
    tt::tt_metal::experimental::inspector::RegisterInspectorRpcChannel(
        "TtnnInspector",
        tt::tt_metal::inspector::rpc::InspectorChannel::Client(
            ::kj::Own<rpc::TtnnInspectorRpcChannel>(&ttnn_inspector_rpc_channel, ::kj::NullDisposer::instance)));
    ttnn_inspector_rpc_channel.setGetMeshWorkloadsCallback(
        [this](auto result) { this->rpc_get_mesh_workloads(result); });
    ttnn_inspector_rpc_channel.setGetMeshWorkloadsRuntimeIdsCallback(
        [this](auto result) { this->rpc_get_mesh_workloads_runtime_ids(result); });
}

void Data::rpc_get_mesh_workloads(rpc::TtnnInspector::GetMeshWorkloadsResults::Builder& results) {
    std::lock_guard<std::mutex> lock(mesh_workloads_mutex);
    auto mesh_workloads = results.initMeshWorkloads(mesh_workloads_data.size());
    uint32_t i = 0;
    for (const auto& [mesh_workload_id, mesh_workload_data] : mesh_workloads_data) {
        auto mesh_workload = mesh_workloads[i++];
        mesh_workload.setMeshWorkloadId(mesh_workload_id);
        mesh_workload.setName(mesh_workload_data.name);
        mesh_workload.setParameters(mesh_workload_data.parameters);
    }
}

void Data::rpc_get_mesh_workloads_runtime_ids(rpc::TtnnInspector::GetMeshWorkloadsRuntimeIdsResults::Builder& results) {
    std::lock_guard<std::mutex> lock(runtime_ids_mutex);
    auto all_runtime_ids = results.initRuntimeIds(runtime_ids.size());
    for (size_t i = 0; i < runtime_ids.size(); ++i) {
        auto entry = all_runtime_ids[i];
        entry.setWorkloadId(runtime_ids[i].workload_id);
        entry.setRuntimeId(runtime_ids[i].runtime_id);
    }
}

}  // namespace ttnn::inspector
