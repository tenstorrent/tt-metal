// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "paged_fill_cache_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>

#include <optional>

namespace ttnn::experimental::prim {

struct PagedFillCacheProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PagedFillCacheParams& operation_attributes,
        const PagedFillCacheInputs& tensor_args,
        Tensor& tensor_return_value);
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
};

}  // namespace ttnn::experimental::prim
