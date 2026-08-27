// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "paged_fill_cache_device_operation_types.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

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

// Still on the legacy ProgramDescriptor concept: this factory's per-coordinate `noop` value differs
// across the mesh, and the Metal 2.0 spec factory concepts have no per-coordinate hook on the
// cache-miss path — create_program_artifacts is called once and one ProgramRunArgs is applied to
// every coordinate.  It therefore keeps building the legacy descriptor, and binds the legacy
// (non-_metal2) kernel sources.
struct PagedFillCacheMeshWorkloadFactory {
    // Per-coord program build.  When mesh_coords is provided and the dispatch
    // coordinate is not in it, the resulting program is a noop (early-exits in
    // kernels) so the cache slot is still populated for that coord.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PagedFillCacheParams& operation_attributes,
        const PagedFillCacheInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const PagedFillCacheParams& operation_attributes,
        const PagedFillCacheInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::experimental::prim
