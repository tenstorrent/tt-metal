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
    // One program per rectangle of coordinates that share `noop`. When mesh_coords is provided,
    // coordinates outside it get a noop program (kernels early-exit) so the cache slot is still
    // populated for them.
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const PagedFillCacheParams& operation_attributes,
        const PagedFillCacheInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    // Called once per coordinate range on a cache hit. Every coordinate of a range this factory
    // emits shares `noop`, so the range's start coordinate decides it.
    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const PagedFillCacheParams& operation_attributes,
        const PagedFillCacheInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRange& range);
};

}  // namespace ttnn::experimental::prim
