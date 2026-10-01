// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "paged_fused_update_cache_device_operation_types.hpp"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/metal_v2_artifacts.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace ttnn::experimental::prim {

struct PagedRowMajorFusedUpdateCacheProgramFactory {
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

    // Single source of truth for the cache_start_id / tile_update_offset_B formulas (shared by
    // create_program_artifacts on a cache miss and override_runtime_arguments on a cache hit). Returns
    // empty in index-tensor mode (positions read on-device).
    static std::vector<PerIndexOffsets> compute_row_major_fused_offsets(
        const PagedFusedUpdateCacheParams& operation_attributes, const PagedFusedUpdateCacheInputs& tensor_args);

    // Cache-hit hook: re-applies every tensor binding plus the hash-excluded cache_start_id /
    // tile_update_offset_B.
    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const PagedFusedUpdateCacheParams& operation_attributes,
        const PagedFusedUpdateCacheInputs& tensor_args,
        PagedFusedUpdateCacheResult& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

struct PagedRowMajorFusedUpdateCacheMeshWorkloadFactory {
    // Per-coord program build. Coordinates outside operation_attributes.mesh_coords (when provided)
    // get no program at all, which is what the ported-from path expressed by handing back an empty
    // ProgramDescriptor for them.
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const PagedFusedUpdateCacheParams& operation_attributes,
        const PagedFusedUpdateCacheInputs& tensor_args,
        PagedFusedUpdateCacheResult& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    // Same program layout as the single-device factory, so it reuses that patch. Only included
    // coordinates have a program, so every range reaching this hook is an included one.
    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const PagedFusedUpdateCacheParams& operation_attributes,
        const PagedFusedUpdateCacheInputs& tensor_args,
        PagedFusedUpdateCacheResult& tensor_return_value,
        const ttnn::MeshCoordinateRange& range);
};

}  // namespace ttnn::experimental::prim
