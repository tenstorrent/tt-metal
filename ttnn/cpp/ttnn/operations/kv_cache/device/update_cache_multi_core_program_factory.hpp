// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <vector>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "update_cache_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

// Per-core runtime args for UPDATE, derived purely from (operation_attributes, tensor_args).
// SINGLE SOURCE OF TRUTH for both create_descriptor() (cache miss, emits the full vectors) and
// get_dynamic_runtime_args() (cache hit, re-applies only the address/attribute-derived slots
// listed in reader_dynamic_indices / writer_dynamic_indices). Address slots already hold the live
// buffer address. There are no noop cores: every entry in `cores` is a work core. The compute
// kernel(s) carry no dynamic args.
struct UpdateCachePerCoreArgs {
    std::vector<tt::tt_metal::CoreCoord> cores;
    std::vector<tt::tt_metal::KernelDescriptor::CoreRuntimeArgs> reader_args;  // indexed by `cores`
    std::vector<tt::tt_metal::KernelDescriptor::CoreRuntimeArgs> writer_args;
    // Arg indices re-applied on every cache hit (buffer addresses + update_idx/batch_offset-derived).
    std::vector<uint32_t> reader_dynamic_indices;  // reader arg 0 = dst, 1 = src, 8 = cache_start_id
    std::vector<uint32_t> writer_dynamic_indices;  // writer arg 0 = dst, 7 = cache_start_id,
                                                   // 10 = tile_update_offset, 11 = batch_read_offset
};

UpdateCachePerCoreArgs compute_update_cache_per_core_args(
    const KvCacheParams& operation_attributes, const KvCacheInputs& tensor_args);

// Descriptor-based factory: builds a ProgramDescriptor declaratively. The framework owns
// program construction, caching, dynamic CB address patching (via CBDescriptor::buffer)
// and runtime arg copy on cache hits, so this struct no longer needs shared_variables_t,
// cached_program_t, create() or override_runtime_arguments().
struct UpdateCacheMultiCoreProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const KvCacheParams& operation_attributes, const KvCacheInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
