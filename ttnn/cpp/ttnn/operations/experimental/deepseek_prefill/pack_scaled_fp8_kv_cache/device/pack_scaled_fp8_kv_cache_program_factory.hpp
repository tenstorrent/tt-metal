// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pack_scaled_fp8_kv_cache_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim::pack_scaled_fp8_kv_cache {

struct PackScaledFp8KvCacheSharedVariables {
    tt::tt_metal::KernelHandle kernel_id = 0;
    std::vector<CoreCoord> cores;
};

struct PackScaledFp8KvCacheProgramFactory {
    using shared_variables_t = PackScaledFp8KvCacheSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const PackScaledFp8KvCacheParams& operation_attributes,
        const PackScaledFp8KvCacheInputs& tensor_args,
        Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const PackScaledFp8KvCacheParams& operation_attributes,
        const PackScaledFp8KvCacheInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::experimental::prim::pack_scaled_fp8_kv_cache
