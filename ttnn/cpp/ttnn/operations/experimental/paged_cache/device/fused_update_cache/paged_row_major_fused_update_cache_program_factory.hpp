// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "paged_fused_update_cache_device_operation_types.hpp"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

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

    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PagedFusedUpdateCacheParams& operation_attributes,
        const PagedFusedUpdateCacheInputs& tensor_args,
        PagedFusedUpdateCacheResult& tensor_return_value);

    // Single source of truth for the cache_start_id / tile_update_offset_B formulas (shared by
    // create_descriptor, which override_runtime_arguments re-runs on cache hits). Returns empty in
    // index-tensor mode (positions read on-device).
    static std::vector<PerIndexOffsets> compute_row_major_fused_offsets(
        const PagedFusedUpdateCacheParams& operation_attributes, const PagedFusedUpdateCacheInputs& tensor_args);
};

struct PagedRowMajorFusedUpdateCacheMeshWorkloadFactory {
    // Per-coord program build.  Coordinates outside operation_attributes.mesh_coords
    // (when provided) get an empty program — the legacy mesh path skipped them
    // entirely; with the descriptor framework we still must hand back a descriptor
    // for every dispatched coord, so we return an empty one for excluded coords.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PagedFusedUpdateCacheParams& operation_attributes,
        const PagedFusedUpdateCacheInputs& tensor_args,
        PagedFusedUpdateCacheResult& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::experimental::prim
