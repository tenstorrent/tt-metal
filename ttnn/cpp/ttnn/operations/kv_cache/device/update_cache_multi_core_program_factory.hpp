// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
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
struct UpdateCacheMultiCoreProgramFactory {
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

// Runtime-arg values that derive from operation_attributes (update_idx, batch_offset) which
// UpdateKVCacheOperation::compute_program_hash deliberately EXCLUDES from the program-cache key
// (so two updates that differ only in those cache-hit), yet are baked into reader/writer runtime
// args. Single source of truth for the work-split + formulas: create_descriptor (cache miss) and
// override_runtime_arguments (cache hit) both call it, so the core order and the arg values cannot
// drift between the two.
//   - cache_start_ids: per-core, in the SAME core order create_descriptor emits runtime args.
//   - tile_update_offset / batch_read_offset: identical on every core (op-wide scalars).
//   - Wbytes: op-wide; not attribute-derived, but it depends on fp32_dest_acc_en, which the hash
//     excludes along with the rest of compute_kernel_config, so the override re-applies it too.
struct UpdateCacheDynamicArgs {
    std::vector<std::pair<tt::tt_metal::CoreCoord, uint32_t>> cache_start_ids;
    uint32_t tile_update_offset = 0;
    uint32_t batch_read_offset = 0;
    uint32_t Wbytes = 0;
};

UpdateCacheDynamicArgs compute_update_cache_dynamic_args(
    const KvCacheParams& operation_attributes, const KvCacheInputs& tensor_args);

}  // namespace ttnn::prim
