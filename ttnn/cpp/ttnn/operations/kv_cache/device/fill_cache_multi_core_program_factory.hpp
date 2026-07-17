// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
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
struct FillCacheMultiCoreProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const KvCacheParams& operation_attributes, const KvCacheInputs& tensor_args, Tensor& tensor_return_value);
};

// Per-core writer cache_start_id, in the SAME core order create_descriptor emits runtime args.
// cache_start_id derives from operation_attributes (batch_idx, update_idx) which
// UpdateKVCacheOperation::compute_program_hash deliberately EXCLUDES from the program-cache key
// (so two fills that differ only in those cache-hit), yet it is baked into a writer runtime arg.
// create_descriptor re-derives it from the current attrs on every dispatch (cache miss and, via
// override_runtime_arguments, cache hit), so it never freezes.
std::vector<std::pair<tt::tt_metal::CoreCoord, uint32_t>> compute_fill_cache_start_ids(
    const KvCacheParams& operation_attributes, const KvCacheInputs& tensor_args);

}  // namespace ttnn::prim
