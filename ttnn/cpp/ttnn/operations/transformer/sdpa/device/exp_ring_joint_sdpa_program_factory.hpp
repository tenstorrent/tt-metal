// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstdlib>
#include <optional>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/transformer/sdpa/device/exp_ring_joint_sdpa_device_operation_types.hpp"

namespace ttnn::prim {

// EXPERIMENT (TT_EXP_SDPA_MUX_BOTTOM_ROW): place the fabric MUX kernels on the BOTTOM ROW of the
// user grid instead of the reserved right column. The SDPA grid then keeps every column and gives
// up the bottom row to the MUX kernels; if the remaining row count is odd one more row idles so the
// backward/forward direction split and the termination groups stay uniform (no kernel changes). On
// a Blackhole 13x10 grid that is 13x8 (two rows given up); on the Wormhole 8x9 grid it is 8x8 = 64
// SDPA cores, one more column than the reserved-column layout's 7x8 = 56. NOTE: the env var is not
// part of the program-cache key — use one setting per process. Shared by the factory build, the
// cache-hit patch, and validation so all three derive the same grid.
inline bool exp_sdpa_mux_on_bottom_row() {
    static const bool enabled = std::getenv("TT_EXP_SDPA_MUX_BOTTOM_ROW") != nullptr;
    return enabled;
}

// SDPA worker grid for a user grid: drop the reserved MUX column (default) or the MUX row plus an
// idle row when the remainder is odd (bottom-row placement). Single source of truth for the build
// and the cache-hit runtime-arg patch.
inline CoreCoord exp_sdpa_grid_for_user_grid(const CoreCoord& user_grid) {
    if (!exp_sdpa_mux_on_bottom_row()) {
        return CoreCoord{user_grid.x - 1, user_grid.y};
    }
    const uint32_t rows = user_grid.y - 1;
    return CoreCoord{user_grid.x, rows - (rows % 2)};
}

// EXPERIMENT (TT_EXP_SDPA_MUX_TOP_CLUSTER): keep the reserved-column MUX placement (and therefore
// the exact same SDPA schedule) but cluster the four MUX kernels at the TOP of the column — rows
// 0..3, which sit 1..4 NoC hops from the eth row (all Blackhole eth cores are at physical row 1) —
// instead of the spread rows 0 / mid-1 / mid / last, whose farthest core is ~10 hops away.
// Isolates the MUX->eth distance effect at a fixed schedule.
inline bool exp_sdpa_mux_top_cluster() {
    static const bool enabled = std::getenv("TT_EXP_SDPA_MUX_TOP_CLUSTER") != nullptr;
    return enabled;
}

// EXPERIMENT (TT_EXP_SDPA_Q_GROUPS=G, G >= 1): sequential passes. Passes run pass-outer /
// ring-inner so one Q chunk and one flash state are live per core (the scratch path) instead of one
// per pass, and each head-segment's Q chunks are split into G groups of one chunk per column, walked
// as extra passes; only group 0 forwards K/V over the fabric, later groups re-read the gathered K/V
// from DRAM. This is what lets a shard whose per-core Q volume does not fit L1 (MiniMax-H3 15 s on a
// Wormhole 7x8 grid) build at all. 0 / unset = the original lockstep schedule. NOTE: not part of the
// program-cache key — one setting per process.
inline uint32_t exp_sdpa_q_groups() {
    static const uint32_t groups = [] {
        const char* v = std::getenv("TT_EXP_SDPA_Q_GROUPS");
        return v ? static_cast<uint32_t>(std::atoi(v)) : 0u;
    }();
    return groups;
}
inline bool exp_sdpa_sequential_passes() { return exp_sdpa_q_groups() >= 1; }

// Single source of truth for the DYNAMIC (hash-excluded) global-semaphore address runtime args.
//
// The per-link GlobalSemaphore addresses are excluded from the program-cache key
// (ExpRingJointSDPAParams::attribute_values omits `semaphore`), so two calls that differ only in
// which GlobalSemaphores they pass still cache-hit. That makes the addresses dynamic: the factory
// bakes them for the cache-miss build, and
// ExpRingJointSDPAMeshWorkloadFactory::override_runtime_arguments() re-applies them on every
// dispatch — otherwise a cache hit with a different semaphore set would silently reuse the address
// frozen at the first miss (the frozen-runtime-arg bug).
//
// The kernel indices, per-core arg slots and total arg counts below are the shared reference for
// BOTH the factory's cache-miss bake (build_exp_ring_joint_sdpa_program_descriptor) and the
// cache-hit patch. The patch asserts arg COUNTS, so adding or removing an arg fails loudly; a
// reorder that keeps the count is not detected, so update these slots in lockstep with the bake.
namespace exp_ring_joint_sdpa_dynamic {
// Kernel indices — must match the desc.kernels push order (reader, writer, writer_fabric, compute[, mux]).
inline constexpr uint32_t kReaderKernelIdx = 0;
inline constexpr uint32_t kWriterFabricKernelIdx = 2;
// Per-core reader runtime-arg slot of the first per-link semaphore address; slots
// kReaderSemaphoreArgBase .. +num_links-1 hold args.semaphore[lnk].address().
// Preceded by 8 tensor addresses, 3 Q-work args, 10 chain/forwarding args, and num_links.
inline constexpr uint32_t kReaderSemaphoreArgBase = 22;
// Reader args after the semaphore addresses: ring_size, ring_index, direction, then the
// split-head forwarding dedup descriptor (dedup_role, buddy_injector_x, buddy_injector_y).
inline constexpr uint32_t kReaderTrailingArgCount = 6;
constexpr uint32_t reader_arg_count(uint32_t num_links) {
    return kReaderSemaphoreArgBase + num_links + kReaderTrailingArgCount;
}
// Per-core fabric-writer runtime-arg slot of out_ready_sem_addr (= args.semaphore[link].address()).
// Preceded by 2 output addresses, 3 Q-work args, 3 ring-indexer args, and 17 MUX args.
inline constexpr uint32_t kWriterFabricOutReadySemArg = 25;
// Fabric-writer args after out_ready_sem_addr: injector x/y, num_muxes, mux index, AG Wt/Ht,
// gathered k/v addresses, dedup_skip_forward (split-head forwarding dedup).
inline constexpr uint32_t kWriterFabricArgCount = kWriterFabricOutReadySemArg + 10;
}  // namespace exp_ring_joint_sdpa_dynamic

namespace detail {

struct ExpRingJointSDPADescriptorAdapterOperation {
    using operation_attributes_t = ExpRingJointSDPAParams;
    using tensor_args_t = ExpRingJointSDPAInputs;
    using spec_return_value_t = ExpRingJointSDPAResultSpec;
    using tensor_return_value_t = ExpRingJointSDPAResult;
};

}  // namespace detail

struct ExpRingJointSDPAProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const ExpRingJointSDPAParams& operation_attributes,
        const ExpRingJointSDPAInputs& tensor_args,
        ExpRingJointSDPAResult& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

struct ExpRingJointSDPAMeshWorkloadFactory {
    using descriptor_adapter_t =
        ttnn::device_operation::MeshDeviceOperationAdapter<detail::ExpRingJointSDPADescriptorAdapterOperation>::
            DescriptorMeshWorkloadAdapter<ExpRingJointSDPAProgramFactory>;
    using cached_mesh_workload_t = typename descriptor_adapter_t::cached_mesh_workload_t;

    static cached_mesh_workload_t create_mesh_workload(
        const ExpRingJointSDPAParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const ExpRingJointSDPAInputs& tensor_args,
        ExpRingJointSDPAResult& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const ExpRingJointSDPAParams& operation_attributes,
        const ExpRingJointSDPAInputs& tensor_args,
        ExpRingJointSDPAResult& tensor_return_value);
};

static_assert(ttnn::device_operation::MeshWorkloadFactoryConcept<ExpRingJointSDPAMeshWorkloadFactory>);

}  // namespace ttnn::prim
