// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "paged_update_cache_device_operation_types.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/metal_v2_artifacts.hpp"

#include <optional>

namespace ttnn::experimental::prim {

struct PagedUpdateCacheProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const PagedUpdateCacheParams& operation_attributes,
        const PagedUpdateCacheInputs& tensor_args,
        Tensor& tensor_return_value);

    // Cache-hit re-derivation of all per-dispatch state: every tensor binding (the borrowed
    // input-shard DFB re-resolves from its own binding) and the cache-write offsets derived from
    // update_idxs, which the program hash excludes so decode steps differing only in position
    // cache-hit.
    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const PagedUpdateCacheParams& operation_attributes,
        const PagedUpdateCacheInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

struct PagedUpdateCacheMeshWorkloadFactory {
    // Per-coord program build. A coordinate outside mesh_coords gets no program at all, which is
    // what the ported-from path expressed by returning an empty ProgramDescriptor for it.
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const PagedUpdateCacheParams& operation_attributes,
        const PagedUpdateCacheInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    // Same program body as PagedUpdateCacheProgramFactory, so it reuses that patch. Only coordinates
    // inside mesh_coords have a program, so every range reaching this hook is an included one.
    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const PagedUpdateCacheParams& operation_attributes,
        const PagedUpdateCacheInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRange& range);
};

}  // namespace ttnn::experimental::prim
