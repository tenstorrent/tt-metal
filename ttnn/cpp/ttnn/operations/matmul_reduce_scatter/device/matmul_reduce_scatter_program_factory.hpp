// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "matmul_reduce_scatter_device_operation_types.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::matmul_reduce_scatter {

// The factory pushes its kernels in this order, so a kernel's index in the cached Program is its value here.
namespace kernel_index {
enum : uint32_t { READER, COMPUTE, WRITER, PORT_READER, PORT_SENDER, COUNT };
}  // namespace kernel_index

// One program per mesh coordinate (its chip's position in its group fixes its schedule and its port peers).
//   create_descriptor          cache miss: grid, ports, schedule words, CBs, semaphores, fabric connections;
//   override_runtime_arguments cache hit: the common address slots (A, W, output, staging) and the three
//                              staging-backed CBs (FINISHED, ARRIVALS_FWD, ARRIVALS_BWD). Everything else is
//                              fixed by the key (the attributes, plan included, and the tensor specs) or lives
//                              in the persistent per-mesh resources below.
struct MatmulReduceScatterProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const MatmulReduceScatterParams& attrs,
        const MatmulReduceScatterInputs& tensors,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const MatmulReduceScatterParams& attrs,
        const MatmulReduceScatterInputs& tensors,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

// ---- shared with the entry / device operation -----------------------------------------------------------------

// Persistent per-mesh resources, created once per mesh (every counter / mailbox zero, synchronized before the
// first call). Their addresses are stable for the mesh's life.
struct MeshResources {
    tt::tt_metal::GlobalSemaphore arrivals_fwd;
    tt::tt_metal::GlobalSemaphore arrivals_bwd;
    tt::tt_metal::GlobalSemaphore ready;
    tt::tt_metal::GlobalSemaphore credit;
    Tensor mailbox;                 // uint32, height-sharded over the full grid, kMailboxStride bytes per core
    std::optional<Tensor> staging;  // bf16 tiled, height-sharded over the full grid
    uint32_t staging_tiles = 0;
};

// The mesh's resources (created on first use); its staging is (re)allocated to exactly staging_tiles per core.
MeshResources& mesh_resources(tt::tt_metal::distributed::MeshDevice* mesh, uint32_t staging_tiles);

// The groups along cluster_axis: one vector of coordinates per group, in group-position order.
std::vector<std::vector<ttnn::MeshCoordinate>> build_groups(
    const tt::tt_metal::distributed::MeshShape& mesh_shape, uint32_t cluster_axis);

// Whether fabric_config wraps cluster_axis (a Ring needs it).
bool fabric_config_wraps(tt::tt_fabric::FabricConfig fabric_config, uint32_t cluster_axis);

// The fewest forwarding links over every hop (both ways) of every group; ring adds the wrap hop.
uint32_t available_links(
    const tt::tt_metal::distributed::MeshDevice& mesh,
    const std::vector<std::vector<ttnn::MeshCoordinate>>& groups,
    bool ring);

}  // namespace ttnn::operations::matmul_reduce_scatter
