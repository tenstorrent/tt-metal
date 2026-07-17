// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <utility>
#include <vector>
#include <tt-metalium/program_descriptors.hpp>
#include "update_cache_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

// Descriptor-based factory: builds a ProgramDescriptor declaratively. The framework owns
// program construction, caching, dynamic CB address patching (via CBDescriptor::buffer)
// and runtime arg copy on cache hits, so this struct no longer needs shared_variables_t,
// cached_program_t, create() or override_runtime_arguments().
struct UpdateCacheMultiCoreProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const KvCacheParams& operation_attributes, const KvCacheInputs& tensor_args, Tensor& tensor_return_value);
};

// Runtime-arg values that derive from operation_attributes (update_idx, batch_offset) which
// UpdateKVCacheOperation::compute_program_hash deliberately EXCLUDES from the program-cache key
// (so two updates that differ only in those cache-hit), yet are baked into reader/writer runtime
// args. create_descriptor re-derives them from the current attrs on every dispatch (cache miss and,
// via override_runtime_arguments, cache hit), so they never freeze.
//   - cache_start_ids: per-core, in the SAME core order create_descriptor emits runtime args.
//   - tile_update_offset / batch_read_offset: identical on every core (op-wide scalars).
struct UpdateCacheDynamicArgs {
    std::vector<std::pair<tt::tt_metal::CoreCoord, uint32_t>> cache_start_ids;
    uint32_t tile_update_offset = 0;
    uint32_t batch_read_offset = 0;
};

UpdateCacheDynamicArgs compute_update_cache_dynamic_args(
    const KvCacheParams& operation_attributes, const KvCacheInputs& tensor_args);

}  // namespace ttnn::prim
