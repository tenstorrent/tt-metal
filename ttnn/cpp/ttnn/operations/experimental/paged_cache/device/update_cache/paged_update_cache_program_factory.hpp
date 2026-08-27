// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "paged_update_cache_device_operation_types.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

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

// Still on the legacy ProgramDescriptor concept: this factory returns a DIFFERENT program per mesh
// coordinate (an empty ProgramDescriptor for a coordinate outside operation_attributes.mesh_coords),
// and the Metal 2.0 spec factory concepts have no per-coordinate hook — create_program_artifacts is
// called once and its spec is stamped on every coordinate.  It therefore keeps building the legacy
// descriptor, and binds the legacy (non-_metal2) kernel sources.
struct PagedUpdateCacheMeshWorkloadFactory {
    // Per-coord program build.  See PagedRowMajorFusedUpdateCacheMeshWorkloadFactory.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PagedUpdateCacheParams& operation_attributes,
        const PagedUpdateCacheInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const PagedUpdateCacheParams& operation_attributes,
        const PagedUpdateCacheInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::experimental::prim
