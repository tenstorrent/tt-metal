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
// Open trade-off (to revisit): allocating per call costs host time on every eager call. On a QuietBox 2 (1x4 ring and
// line, a tiny 32x256 @ 256x512 per-device shape, program cache warm) a hot call went from 13.8 to ~35 us of host
// time: creating + freeing a height-sharded L1 tensor over the full grid costs ~24 us there (an L1 interleaved one
// ~4 us). Traced calls pay nothing. The alternatives considered:
//   - one persistent staging per mesh, reallocated when the size changes (the previous scheme): no host cost, but a
//     trace that holds two shapes with different staging sizes points at freed L1;
//   - persistent, one tensor per staging size: no host cost and trace-safe, but holds L1 for every size used;
//   - persistent, grow-only: no host cost, smallest persistent footprint, but a trace stays valid only if the largest
//     shape ran before capture.
// A cheaper per-call allocation (e.g. a buffer without the sharded page mapping) would remove the cost without the
// persistent L1.
Tensor allocate_staging(tt::tt_metal::distributed::MeshDevice* mesh, uint32_t staging_tiles);
// Lookup only (factory and the hit path); TT_FATAL when the entry has not created them.
MeshResources& mesh_resources(tt::tt_metal::distributed::MeshDevice* mesh);

// The last call's plan (the binding exposes it as a debugging aid).
void record_last_plan(const MatmulReduceScatterPrefillParams& attrs);
const std::optional<MatmulReduceScatterPrefillParams>& last_plan();

}  // namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_prefill
