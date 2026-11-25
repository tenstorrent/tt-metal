// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/global_semaphore.hpp>

#include "ttnn/distributed/types.hpp"
#include "all_gather_rms_norm_device_operation_types.hpp"
#include "all_gather_rms_norm_planner.hpp"

namespace ttnn::operations::all_gather_rms_norm {

// Persistent per-mesh state (Python global_semaphores): NUM_GLOBAL_SEMS global semaphores on the port cores, created
// once per (mesh id, shape, port set) on the first call, before the call allocates any L1 tensor; held in a
// function-local ttsl::Indestructible map. Addresses are stable for the mesh's life (never rewritten on a hit).
struct PersistentSems {
    std::array<uint32_t, planner::NUM_GLOBAL_SEMS> addresses{};  // ready, bulk, stats, bands..., stage counter
    uint32_t ready() const { return addresses[0]; }
    uint32_t bulk() const { return addresses[1]; }
    uint32_t stats() const { return addresses[2]; }
    uint32_t band(uint32_t b) const { return addresses[3 + b]; }
    uint32_t stage_counter() const { return addresses[planner::NUM_GLOBAL_SEMS - 1]; }
};
const PersistentSems& persistent_semaphores(
    tt::tt_metal::distributed::MeshDevice& mesh, const planner::MeshPlan& mesh_plan);

// One program per mesh coordinate. create_descriptor plans once on a miss; override_runtime_arguments writes the
// tensor addresses (worker common X/DST/OUT/GAMMA/BETA, port common DST_ADDR), the shard-backed CB addresses
// (cb::X, cb::X_STATS, cb::OUT_BLOCK) and EPS_BITS, only when they changed since the last hit.
struct AllGatherRmsNormProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const AllGatherRmsNormParams& attrs,
        const AllGatherRmsNormInputs& tensors,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const AllGatherRmsNormParams& attrs,
        const AllGatherRmsNormInputs& tensors,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::operations::all_gather_rms_norm
