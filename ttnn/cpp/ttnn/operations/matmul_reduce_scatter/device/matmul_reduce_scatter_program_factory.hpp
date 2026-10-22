// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <optional>

#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "matmul_reduce_scatter_device_operation_types.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::matmul_reduce_scatter {

// The global semaphores of a mesh, over the full worker grid; COUNT is how many there are.
enum class GlobalSem : uint32_t { ARR_FWD, ARR_BWD, READY, CREDIT, COUNT };

// Per-mesh persistent state, created once per mesh (global semaphores and mailbox zeroed and the mesh
// synchronized at creation) and kept alive for the process. The staging tensor is reallocated whenever a call
// needs a different size (the Python's "exact" policy).
struct MeshResources {
    tt::tt_metal::distributed::MeshDevice* mesh = nullptr;
    std::array<std::optional<tt::tt_metal::GlobalSemaphore>, static_cast<size_t>(GlobalSem::COUNT)> semaphores;
    std::optional<Tensor> mailbox;  // uint32 row-major, height-sharded over the grid, MAILBOX_STRIDE bytes per core
    std::optional<Tensor> staging;  // bf16 tiled, height-sharded over the grid, staging_tiles tiles per core
    uint32_t staging_tiles = 0;
};

// Creates the mesh's resources on first use and sizes staging to `staging_tiles`; called by the entry.
MeshResources& acquire_mesh_resources(tt::tt_metal::distributed::MeshDevice* mesh, uint32_t staging_tiles);
// The resources already created for `mesh` (the factory and override read the semaphore addresses here).
const MeshResources& resources_of(const tt::tt_metal::distributed::MeshDevice* mesh);

// One program per mesh coordinate: create_descriptor builds that chip's program.
struct MatmulReduceScatterProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const MatmulReduceScatterParams& attrs,
        const MatmulReduceScatterInputs& tensor_args,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const MatmulReduceScatterParams& attrs,
        const MatmulReduceScatterInputs& tensor_args,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::operations::matmul_reduce_scatter
