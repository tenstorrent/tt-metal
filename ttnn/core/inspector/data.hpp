// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <core/inspector/ttnn_rpc.capnp.h>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>

#include "core/inspector/ttnn_rpc_channel_generated.hpp"

namespace ttnn::inspector {

struct MeshWorkloadData {
    uint64_t mesh_workload_id{};
    std::string name;
    std::string parameters;
};

struct MeshWorkloadRuntimeIdEntry {
    uint64_t workload_id = 0;
    uint64_t runtime_id = 0;
};

class Data {
public:
    static Data& instance();

    void register_inspector_rpc();

    void rpc_get_mesh_workloads(rpc::TtnnInspector::GetMeshWorkloadsResults::Builder& results);
    void rpc_get_mesh_workloads_runtime_ids(rpc::TtnnInspector::GetMeshWorkloadsRuntimeIdsResults::Builder& results);

    std::mutex mesh_workloads_mutex;
    std::unordered_map<uint64_t, MeshWorkloadData> mesh_workloads_data;

    std::mutex runtime_ids_mutex;
    std::deque<MeshWorkloadRuntimeIdEntry> runtime_ids;
    static constexpr size_t MAX_RUNTIME_ID_ENTRIES = 10000;

    rpc::TtnnInspectorRpcChannel ttnn_inspector_rpc_channel;
};

}  // namespace ttnn::inspector
