// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "paged_update_cache_device_operation_types.hpp"

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include <optional>

namespace ttnn::experimental::prim {

struct PagedUpdateCacheProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PagedUpdateCacheParams& operation_attributes,
        const PagedUpdateCacheInputs& tensor_args,
        Tensor& tensor_return_value);

    // Cache-hit in-place patch of all per-dispatch state: buffer addresses (runtime args + the input-shard
    // CB) and the cache-write offsets derived from update_idxs, which the program hash excludes so decode
    // steps differing only in position cache-hit. No descriptor rebuild — see the .cpp.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const PagedUpdateCacheParams& operation_attributes,
        const PagedUpdateCacheInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

struct PagedUpdateCacheMeshWorkloadFactory {
    // Per-coord program build.  See PagedRowMajorFusedUpdateCacheMeshWorkloadFactory.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PagedUpdateCacheParams& operation_attributes,
        const PagedUpdateCacheInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);

    // Same program body as PagedUpdateCacheProgramFactory, so it reuses that patch.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const PagedUpdateCacheParams& operation_attributes,
        const PagedUpdateCacheInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::experimental::prim
