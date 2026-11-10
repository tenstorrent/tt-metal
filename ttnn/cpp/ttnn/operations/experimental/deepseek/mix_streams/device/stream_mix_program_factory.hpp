// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "stream_mix_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek::mix_streams {

struct StreamMixSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    std::vector<CoreCoord> cores;
};

struct StreamMixProgramFactory {
    using shared_variables_t = StreamMixSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const StreamMixParams& operation_attributes,
        const StreamMixInputs& tensor_args,
        StreamMixTensorReturn& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const StreamMixParams& operation_attributes,
        const StreamMixInputs& tensor_args,
        StreamMixTensorReturn& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deepseek::mix_streams
