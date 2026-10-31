// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <functional>
#include <optional>

#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "matmul_reduce_scatter_prefill_device_operation_types.hpp"

namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_prefill {

// One program per mesh coordinate. Kernels in
// ::matmul_reduce_scatter_prefill::kernel_index order: reader, compute, writer, port reader, port sender.
struct MatmulReduceScatterPrefillProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const MatmulReduceScatterPrefillParams& attrs,
        const MatmulReduceScatterPrefillInputs& tensor_args,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

// The mesh workload: a program per coordinate from create_descriptor, with what a cache hit writes kept at creation
// (the four kernels' common-arg objects, the three staging-backed CBs and their zone offsets), so a hit computes the
// call's addresses once and writes them into each program, with no lookup per program.
struct MatmulReduceScatterPrefillMeshWorkloadFactory {
    struct shared_variables_t {
        // Cache the argument objects, not their data(): dispatch may relocate the payload.
        std::reference_wrapper<tt::tt_metal::RuntimeArgsData> reader;
        std::reference_wrapper<tt::tt_metal::RuntimeArgsData> writer;
        std::reference_wrapper<tt::tt_metal::RuntimeArgsData> port_reader;
        std::reference_wrapper<tt::tt_metal::RuntimeArgsData> port_sender;
        std::array<tt::tt_metal::CBHandle, 3> staging_cbs{};  // FINISHED, ARRIVALS_FWD, ARRIVALS_BWD
        std::array<uint32_t, 3> staging_offsets{};
    };
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const MatmulReduceScatterPrefillParams& attrs,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const MatmulReduceScatterPrefillInputs& tensor_args,
        Tensor& output);

    // A, W and the output on every hit; the staging address and its CBs only when the staging tensor moved; the
    // per-mesh semaphores and mailbox never (they keep create_descriptor's values for the mesh's life).
    static void override_runtime_arguments(
        cached_mesh_workload_t& cached,
        const MatmulReduceScatterPrefillParams& attrs,
        const MatmulReduceScatterPrefillInputs& tensor_args,
        Tensor& output);
};

// Per-mesh persistent state: 4 global semaphores (arr_fwd, arr_bwd, ready, credit) on the full grid and the replicated
// uint32 mailbox (height-sharded, (1, n * kMailboxStride / 4) per core); the kernels return every counter to 0, so
// they carry over between calls. Held in a function-local ttsl::Indestructible map keyed by the mesh id, behind a
// mutex. Never released: the map outlives every mesh, so a closed mesh's entry stays until the process exits.
struct MeshResources {
    std::array<tt::tt_metal::GlobalSemaphore, 4> sems;
    Tensor mailbox;
};
enum MeshSem : uint32_t { ARR_FWD, ARR_BWD, READY, CREDIT };

// Called by the public entry before launch: creates the mesh's resources on first use (synchronizing the device once).
// The first call cannot be traced: a mesh with no resources yet refuses a call made during trace capture.
MeshResources& ensure_mesh_resources(tt::tt_metal::distributed::MeshDevice* mesh);
// One call's staging tensor (bf16 tiles, height-sharded over the full grid, staging_tiles tiles per core). The mesh
// allocator gives it the same address on every chip, which the fabric writes rely on; the entry frees it when the call
// has been enqueued: a neighbour writes into it only after this chip's fence_ready, and this chip's program finishes
// only after every arrival has landed.
//
// Per call by design: the op holds no L1 between calls. The layout of a size (tensor spec, sharded page mapping over
// the full grid, topology) is built on its first call and kept per mesh and size, host-side only and never released,
// so a later call only allocates the buffer and wraps it. Costs:
//   - host time on every eager call: allocating and freeing that buffer. Traced replays pay nothing.
//   - traces: the staging is freed during capture, so an L1 buffer allocated after capture can land on it and is
//     overwritten by every replay (tt-metal's general rule: buffers allocated while a trace exists are unsafe unless
//     released before it runs).
// Persistent staging (per mesh, per size, or grow-only) avoids both but holds 170-362 KiB per core for the life of the
// mesh.
Tensor allocate_staging(tt::tt_metal::distributed::MeshDevice* mesh, uint32_t staging_tiles);
// Lookup only (factory and the hit path); TT_FATAL when the entry has not created them.
MeshResources& mesh_resources(tt::tt_metal::distributed::MeshDevice* mesh);

// The last call's plan (the binding exposes it as a debugging aid).
void record_last_plan(const MatmulReduceScatterPrefillParams& attrs);
const std::optional<MatmulReduceScatterPrefillParams>& last_plan();

}  // namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_prefill
