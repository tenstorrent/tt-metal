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

    // Rewrites the A / W / output addresses, and the staging address and tensor-backed CBs 3, 4, 5 when it moved.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const MatmulReduceScatterPrefillParams& attrs,
        const MatmulReduceScatterPrefillInputs& tensor_args,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
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
// One call's staging tensor (bf16 tiles, height-sharded over the full grid, staging_tiles tiles per core). The mesh
// allocator gives it the same address on every chip, which the fabric writes rely on; the entry frees it when the call
// has been enqueued: a neighbour writes into it only after this chip's fence_ready, and this chip's program finishes
// only after every arrival has landed.
//
// Per call by design: the op holds no L1 between calls. Costs:
//   - host time on every eager call: creating + freeing a height-sharded L1 tensor over the full grid (~21 us on a
//     QuietBox 2; a hot call 17 -> 34-36 us host, invisible end to end once the device time exceeds that). Traced
//     replays pay nothing.
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
