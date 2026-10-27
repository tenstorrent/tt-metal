// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>

#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

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

    // Rewrites A / W / output / staging / mailbox / semaphore addresses (common args, tensor-backed CBs 3, 4, 5).
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const MatmulReduceScatterPrefillParams& attrs,
        const MatmulReduceScatterPrefillInputs& tensor_args,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

// Per-mesh persistent state: 4 global semaphores (arr_fwd, arr_bwd, ready, credit) on the
// full grid, the replicated uint32 mailbox (height-sharded, (1, n * kMailboxStride / 4) per core) and the staging
// tensor (bf16 tiles, height-sharded, staging_tiles tiles per core, reallocated whenever its size differs: "exact").
// Held in a function-local ttsl::Indestructible map keyed by the MeshDevice*.
struct MeshResources {
    std::array<tt::tt_metal::GlobalSemaphore, 4> sems;
    Tensor mailbox;
    std::optional<Tensor> staging;
    uint32_t staging_tiles = 0;
};
enum MeshSem : uint32_t { ARR_FWD, ARR_BWD, READY, CREDIT };

// Called by the public entry before launch: creates the mesh's resources on first use (synchronizing the device once)
// and sizes its staging tensor to `staging_tiles`.
MeshResources& ensure_mesh_resources(tt::tt_metal::distributed::MeshDevice* mesh, uint32_t staging_tiles);
// Lookup only (factory and the hit path); TT_FATAL when the entry has not created them.
MeshResources& mesh_resources(tt::tt_metal::distributed::MeshDevice* mesh);

// The last call's plan (the binding exposes it as a debugging aid).
void record_last_plan(const MatmulReduceScatterPrefillParams& attrs);
const std::optional<MatmulReduceScatterPrefillParams>& last_plan();

}  // namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_prefill
