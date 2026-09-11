// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "paged_update_cache_device_operation_types.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/program.hpp>

#include "ttnn/metal_v2_artifacts.hpp"

#include <optional>

namespace ttnn::experimental::prim {

// Metal 2.0 factory (CustomProgramSpecFactoryConcept).  Selected when mesh_coords is nullopt.
struct PagedUpdateCacheProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const PagedUpdateCacheParams& operation_attributes,
        const PagedUpdateCacheInputs& tensor_args,
        Tensor& tensor_return_value);

    // Cache-hit re-derivation of all per-dispatch state: the tensor bindings (which on this concept
    // the framework does NOT refresh for us — including the borrowed-memory input DFB's backing
    // address) and the cache-write offsets derived from update_idxs, which the program hash excludes
    // so decode steps differing only in position cache-hit.  See the .cpp.
    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const PagedUpdateCacheParams& operation_attributes,
        const PagedUpdateCacheInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

// Metal 2.0 mesh-workload factory (MeshWorkloadSpecFactoryConcept).  Selected when mesh_coords is
// provided.  Distinct from its single-device sibling in exactly one way: it runs a different set of
// programs across the mesh, because a coordinate outside operation_attributes.mesh_coords gets no
// program at all.  Expressing that needs a per-coordinate artifact, which is what this concept adds.
struct PagedUpdateCacheMeshWorkloadFactory {
    // One ProgramSpec + ProgramRunArgs per coordinate range.  Ranges omitted from the result get no
    // program, which is how the coordinate filter is expressed.
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const PagedUpdateCacheParams& operation_attributes,
        const PagedUpdateCacheInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    // Cache-hit refresh, called once per range (not once per device).  Same per-dispatch state as the
    // single-device factory refreshes; see the .cpp for why the range is not needed.
    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const PagedUpdateCacheParams& operation_attributes,
        const PagedUpdateCacheInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRange& coordinate_range);
};

}  // namespace ttnn::experimental::prim
