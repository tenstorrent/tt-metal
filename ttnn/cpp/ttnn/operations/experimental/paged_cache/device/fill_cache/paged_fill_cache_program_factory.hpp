// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "paged_fill_cache_device_operation_types.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/program.hpp>

#include "ttnn/metal_v2_artifacts.hpp"

#include <optional>

namespace ttnn::experimental::prim {

// Metal 2.0 factory (CustomProgramSpecFactoryConcept).  Selected when mesh_coords is nullopt.
struct PagedFillCacheProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const PagedFillCacheParams& operation_attributes,
        const PagedFillCacheInputs& tensor_args,
        Tensor& tensor_return_value);

    // Cache-hit re-derivation. On this concept the framework refreshes nothing on our behalf, so
    // this re-applies every tensor binding plus the args derived from what compute_program_hash
    // excludes — batch_idx_fallback and noop — which would otherwise freeze at the cache-miss value.
    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const PagedFillCacheParams& operation_attributes,
        const PagedFillCacheInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

// Metal 2.0 mesh-workload factory (MeshWorkloadSpecFactoryConcept).  Selected when mesh_coords is
// provided.  Every coordinate gets a program here -- an excluded one gets a *noop* program whose
// kernels early-exit, so its cache slot is still populated.  What this concept supplies is therefore
// per-coordinate run args on the cache miss, not per-coordinate programs.
struct PagedFillCacheMeshWorkloadFactory {
    // One ProgramSpec + ProgramRunArgs per coordinate.  The spec is identical across the mesh; only
    // the `noop` runtime arg differs.
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const PagedFillCacheParams& operation_attributes,
        const PagedFillCacheInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    // Cache-hit refresh, called once per range (not once per device).  Each range covers one
    // coordinate, so the per-coordinate `noop` it re-derives is exact.
    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const PagedFillCacheParams& operation_attributes,
        const PagedFillCacheInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRange& coordinate_range);
};

}  // namespace ttnn::experimental::prim
