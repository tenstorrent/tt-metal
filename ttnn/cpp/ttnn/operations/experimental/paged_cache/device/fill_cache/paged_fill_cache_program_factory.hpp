// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "paged_fill_cache_device_operation_types.hpp"

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include <optional>

namespace ttnn::experimental::prim {

struct PagedFillCacheProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PagedFillCacheParams& operation_attributes,
        const PagedFillCacheInputs& tensor_args,
        Tensor& tensor_return_value);

    // Cache-hit re-derivation: patches the cached program's runtime args in place (no descriptor
    // rebuild). Re-applies every buffer address plus the args derived from what compute_program_hash
    // excludes — batch_idx_fallback and noop — which would otherwise freeze at the cache-miss value.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const PagedFillCacheParams& operation_attributes,
        const PagedFillCacheInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

struct PagedFillCacheMeshWorkloadFactory {
    // Per-coord program build.  When mesh_coords is provided and the dispatch
    // coordinate is not in it, the resulting program is a noop (early-exits in
    // kernels) so the cache slot is still populated for that coord.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PagedFillCacheParams& operation_attributes,
        const PagedFillCacheInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);

    // Same descriptor layout as PagedFillCacheProgramFactory, so it reuses that patch.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const PagedFillCacheParams& operation_attributes,
        const PagedFillCacheInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::experimental::prim
