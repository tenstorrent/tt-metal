// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "indexed_fused_update_cache_device_operation_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim::indexed_fused_update_cache {

struct IndexedFusedUpdateCacheProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const IndexedFusedUpdateCacheParams& operation_attributes,
        const IndexedFusedUpdateCacheInputs& tensor_args,
        IndexedFusedUpdateCacheResult& output);
};

}  // namespace ttnn::experimental::prim::indexed_fused_update_cache
