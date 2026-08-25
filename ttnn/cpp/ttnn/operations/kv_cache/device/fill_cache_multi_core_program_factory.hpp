// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>
#include <utility>
#include <vector>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "update_cache_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::prim {

// Descriptor-based factory: builds a ProgramDescriptor declaratively. The framework owns
// program construction and caching, so this struct needs no shared_variables_t/cached_program_t/create().
struct FillCacheMultiCoreProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const KvCacheParams& operation_attributes, const KvCacheInputs& tensor_args, Tensor& tensor_return_value);

    // Cache-hit hook: patches the per-dispatch runtime args of the cached program in place. The
    // framework calls it on the factory that actually built the cached program, and uses neither
    // resolve_bindings nor get_dynamic_runtime_args, so this owns every buffer address too.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const KvCacheParams& operation_attributes,
        const KvCacheInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

// Per-core writer cache_start_id, in the SAME core order create_descriptor emits runtime args.
// cache_start_id derives from operation_attributes (batch_idx, update_idx) which
// UpdateKVCacheOperation::compute_program_hash deliberately EXCLUDES from the program-cache key
// (so two fills that differ only in those cache-hit), yet it is baked into a writer runtime arg.
// Single source of truth for the work-split + formula: create_descriptor (cache miss) and
// override_runtime_arguments (cache hit) both call it, so the core order and the arg values cannot
// drift between the two.
std::vector<std::pair<tt::tt_metal::CoreCoord, uint32_t>> compute_fill_cache_start_ids(
    const KvCacheParams& operation_attributes, const KvCacheInputs& tensor_args);

}  // namespace ttnn::prim
