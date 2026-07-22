// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "all_gather_regime_a_matmul_async_device_operation_types.hpp"

namespace ttnn::experimental::prim {

// Phase A (DRAM-staged) fused all-gather + regime_a_matmul program factory.
// Task 3 implements create(); the D=1 path never reaches here (the public op delegates to regime_a_matmul).
struct AllGatherRegimeAMatmulAsyncProgramFactory {
    struct shared_variables_t {
        // Populated by create() in Task 3 (semaphore ids, gather-buffer address, kernel handles) and used by
        // override_runtime_arguments() on program-cache replay.
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const AllGatherRegimeAMatmulAsyncParams& operation_attributes,
        const AllGatherRegimeAMatmulAsyncInputs& tensor_args,
        std::vector<Tensor>& output_tensors);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const AllGatherRegimeAMatmulAsyncParams& operation_attributes,
        const AllGatherRegimeAMatmulAsyncInputs& tensor_args,
        std::vector<Tensor>& output_tensors);
};

}  // namespace ttnn::experimental::prim
