// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "paged_fill_cache_device_operation_types.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/metal_v2_artifacts.hpp"

#include <optional>

namespace ttnn::experimental::prim {

struct PagedFillCacheProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const PagedFillCacheParams& operation_attributes,
        const PagedFillCacheInputs& tensor_args,
        Tensor& tensor_return_value);

    // Cache-hit re-derivation. Re-applies every tensor binding plus the args derived from what
    // compute_program_hash excludes — batch_idx_fallback and noop — which would otherwise freeze at
    // the cache-miss value.
    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const PagedFillCacheParams& operation_attributes,
        const PagedFillCacheInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

struct PagedFillCacheMeshWorkloadFactory {
    // Per-coord program build. When mesh_coords is provided and the dispatch coordinate is not in
    // it, the resulting program is a noop (early-exits in kernels) so the cache slot is still
    // populated for that coord.
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const PagedFillCacheParams& operation_attributes,
        const PagedFillCacheInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    // Called once per coordinate range on a cache hit. Every range this factory emits covers a
    // single coordinate, so the range's start coordinate decides `noop`.
    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const PagedFillCacheParams& operation_attributes,
        const PagedFillCacheInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRange& range);
};

}  // namespace ttnn::experimental::prim
