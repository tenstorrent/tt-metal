// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string_view>

#include <tt-metalium/mesh_workload.hpp>

namespace ttnn {
namespace inspector {

/**
 * @brief Registers the TTNN Inspector RPC channel with the Inspector RPC server.
 *
 * This function enables TTNN runtime state to be queried through the Inspector interface.
 * It should be called during system or application initialization, after the Inspector RPC server
 * has been started and is ready to accept channel registrations.
 *
 * Prerequisites:
 *   - The Inspector RPC server must be running and accessible.
 *   - Any required TTNN runtime components should be initialized prior to calling this function.
 */
void register_inspector_rpc();

// Inspector-only annotation for correlating higher-level execution with MeshWorkload runs.
void EmitMeshWorkloadAnnotation(
    tt::tt_metal::distributed::MeshWorkload& workload,
    std::string_view operation_name,
    std::string_view operation_parameters);

// Inspector-only runtime id for correlating workload enqueues/runs across tools.
void EmitMeshWorkloadRuntimeId(tt::tt_metal::distributed::MeshWorkload& workload, uint64_t runtime_id);

}  // namespace inspector
}  // namespace ttnn
