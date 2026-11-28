// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <optional>

#include <tt-metalium/workload/legacy/circular_buffer.hpp>
#include <tt-metalium/workload/program.hpp>
#include <tt-metalium/utils/program_cache.hpp>
#include <tt-metalium/workload/legacy/runtime_args_data.hpp>
#include <tt-metalium/workload/legacy/program_descriptors.hpp>
#include <tt-metalium/memory/cross_program/global_semaphore.hpp>

#include "ttnn/distributed/types.hpp"
#include "all_gather_rms_norm_device_operation_types.hpp"
#include "all_gather_rms_norm_planner.hpp"

namespace ttnn::operations::experimental::ccl::all_gather_rms_norm {

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

// One program per mesh coordinate; create_descriptor plans once on a miss.
struct AllGatherRmsNormProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const AllGatherRmsNormParams& attrs,
        const AllGatherRmsNormInputs& tensors,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

// Owns the mesh workload: every program comes from AllGatherRmsNormProgramFactory::create_descriptor. A hit reads
// the call's tensor addresses and EPS_BITS once and, per program, writes the worker common X/DST/OUT/GAMMA/BETA and
// EPS_BITS, the port common DST_ADDR and the shard-backed CB addresses (cb::X, cb::X_STATS, cb::OUT_BLOCK) only when
// they differ from what the last completed hit (or the miss) wrote.
struct AllGatherRmsNormMeshWorkloadFactory {
    using descriptor_factory_t = AllGatherRmsNormProgramFactory;

    // what a hit writes: the call's tensor addresses and epsilon bits
    struct written_t {
        uint32_t in_addr = 0;
        uint32_t dst_addr = 0;
        uint32_t out_addr = 0;
        uint32_t gamma_addr = 0;
        uint32_t beta_addr = 0;
        uint32_t eps_bits = 0;
        bool operator==(const written_t&) const = default;
    };

    struct shared_variables_t {
        std::array<std::reference_wrapper<tt::tt_metal::RuntimeArgsData>, 3> worker_common;  // READER, WRITER, COMPUTE
        std::reference_wrapper<tt::tt_metal::RuntimeArgsData> port_reader_common;
        std::reference_wrapper<tt::tt_metal::RuntimeArgsData> port_sender_common;
        std::optional<tt::tt_metal::CBHandle> x_cb;          // cb::X backed by the input shard
        std::optional<tt::tt_metal::CBHandle> x_stats_cb;    // cb::X_STATS, the second view of the input shard
        std::optional<tt::tt_metal::CBHandle> out_block_cb;  // cb::OUT_BLOCK backed by the output shard
        uint32_t out_block_offset = 0;                       // this chip's slice of the output shard
        std::optional<written_t> last_written;               // cleared before the first write, set after the last
    };
    using cached_mesh_workload_t = tt::tt_metal::program_cache::detail::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const AllGatherRmsNormParams& attrs,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const AllGatherRmsNormInputs& tensors,
        Tensor& output);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const AllGatherRmsNormParams& attrs,
        const AllGatherRmsNormInputs& tensors,
        Tensor& output);
};

}  // namespace ttnn::operations::experimental::ccl::all_gather_rms_norm
