// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dram_prefetcher_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

// DRAM-core program factory: runs the prefetcher as a DRISC kernel on programmable DRAM cores.
// Mirrors DramPrefetcherProgramFactory's interface but skips the sender-side worker CBs (no c_0,
// no c_3) — the DRISC kernel manages its own 2-stage scratch in DRISC L1 and hand-populates the
// remote sender CB interface from runtime args.
struct DramPrefetcherDramCoreProgramFactory {
    struct shared_variables_t {};

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const DramPrefetcherParams& operation_attributes,
        const DramPrefetcherInputs& tensor_args,
        Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const DramPrefetcherParams& operation_attributes,
        const DramPrefetcherInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::prim
