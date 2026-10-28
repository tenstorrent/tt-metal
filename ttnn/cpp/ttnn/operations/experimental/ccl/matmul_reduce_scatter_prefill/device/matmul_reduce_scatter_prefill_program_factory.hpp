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
// (the reader's and writer's common-arg objects), so a hit computes the call's addresses once and writes them into
// each program, with no lookup per program.
struct MatmulReduceScatterPrefillMeshWorkloadFactory {
    struct shared_variables_t {
        // Cache the argument objects, not their data(): dispatch may relocate the payload.
        std::reference_wrapper<tt::tt_metal::RuntimeArgsData> reader;
        std::reference_wrapper<tt::tt_metal::RuntimeArgsData> writer;
    };
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const MatmulReduceScatterPrefillParams& attrs,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const MatmulReduceScatterPrefillInputs& tensor_args,
        Tensor& output);

    // A, W and the output, on every hit. Staging is the programs' own circular buffers; the per-mesh semaphores and
    // mailbox keep create_descriptor's values for the mesh's life.
    static void override_runtime_arguments(
        cached_mesh_workload_t& cached,
        const MatmulReduceScatterPrefillParams& attrs,
        const MatmulReduceScatterPrefillInputs& tensor_args,
        Tensor& output);
};

// Per-mesh persistent state: 4 global semaphores (arr_fwd, arr_bwd, ready, credit) on the full grid and the replicated
// uint32 mailbox (height-sharded, (1, n * kMailboxStride / 4) per core); the kernels return every counter to 0, so
// they carry over between calls. Held in a function-local ttsl::Indestructible map keyed by the mesh id.
struct MeshResources {
    std::array<tt::tt_metal::GlobalSemaphore, 4> sems;
    Tensor mailbox;
};
enum MeshSem : uint32_t { ARR_FWD, ARR_BWD, READY, CREDIT };

// Called by the public entry before launch: creates the mesh's resources on first use (synchronizing the device once).
MeshResources& ensure_mesh_resources(tt::tt_metal::distributed::MeshDevice* mesh);
// Lookup only (factory and the hit path); TT_FATAL when the entry has not created them.
MeshResources& mesh_resources(tt::tt_metal::distributed::MeshDevice* mesh);

// The last call's plan (the binding exposes it as a debugging aid).
void record_last_plan(const MatmulReduceScatterPrefillParams& attrs);
const std::optional<MatmulReduceScatterPrefillParams>& last_plan();

}  // namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_prefill
