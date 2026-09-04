// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "paged_fused_update_cache_device_operation_types.hpp"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/metal_v2_artifacts.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace ttnn::experimental::prim {

// Metal 2.0 factory (CustomProgramSpecFactoryConcept).  Selected when mesh_coords is nullopt.
struct PagedTiledFusedUpdateCacheProgramFactory {
    // Per-index cache-write offsets derived from update_idxs. One entry per index i over cores1.size();
    // each handles input1 on core1 and input2 on core2, both sharing the same offsets.
    struct PerIndexOffsets {
        tt::tt_metal::CoreCoord core1;
        tt::tt_metal::CoreCoord core2;
        uint32_t cache_start_id = 0;
        uint32_t tile_update_offset_B = 0;
    };

    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const PagedFusedUpdateCacheParams& operation_attributes,
        const PagedFusedUpdateCacheInputs& tensor_args,
        PagedFusedUpdateCacheResult& tensor_return_value);

    // Single source of truth for the cache_start_id / tile_update_offset_B formulas, shared by three
    // callers that must not drift: create_program_artifacts on a cache miss, override_runtime_arguments
    // on a cache hit, and the retained legacy descriptor body the mesh factory still builds. Returns
    // empty in index-tensor mode (positions read on-device).
    static std::vector<PerIndexOffsets> compute_tiled_fused_offsets(
        const PagedFusedUpdateCacheParams& operation_attributes, const PagedFusedUpdateCacheInputs& tensor_args);

    // Cache-hit re-derivation of all per-dispatch state: the tensor bindings (which on this concept
    // the framework does NOT refresh for us -- including the borrowed-memory input DFBs' backing
    // addresses) and the cache-write offsets derived from update_idxs, which the program hash excludes
    // so decode steps differing only in position cache-hit.  See the .cpp.
    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const PagedFusedUpdateCacheParams& operation_attributes,
        const PagedFusedUpdateCacheInputs& tensor_args,
        PagedFusedUpdateCacheResult& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

struct PagedTiledFusedUpdateCacheMeshWorkloadFactory {
    // Per-coord program build.  Coordinates outside operation_attributes.mesh_coords
    // (when provided) get an empty program — the legacy mesh path skipped them
    // entirely; with the descriptor framework we still must hand back a descriptor
    // for every dispatched coord, so we return an empty one for excluded coords.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PagedFusedUpdateCacheParams& operation_attributes,
        const PagedFusedUpdateCacheInputs& tensor_args,
        PagedFusedUpdateCacheResult& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);

    // Same program layout as the single-device factory, so it reuses that patch.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const PagedFusedUpdateCacheParams& operation_attributes,
        const PagedFusedUpdateCacheInputs& tensor_args,
        PagedFusedUpdateCacheResult& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::experimental::prim
