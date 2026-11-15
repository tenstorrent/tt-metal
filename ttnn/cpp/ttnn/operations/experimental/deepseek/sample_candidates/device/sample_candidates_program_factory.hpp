// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "sample_candidates_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek::sample_candidates {

struct SampleCandidatesSharedVariables {
    tt::tt_metal::KernelHandle kernel_id = 0;
    std::vector<CoreCoord> cores;
};

struct SampleCandidatesProgramFactory {
    using shared_variables_t = SampleCandidatesSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const SampleCandidatesParams& operation_attributes,
        const SampleCandidatesInputs& tensor_args,
        SampleCandidatesTensorReturn& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const SampleCandidatesParams& operation_attributes,
        const SampleCandidatesInputs& tensor_args,
        SampleCandidatesTensorReturn& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deepseek::sample_candidates
