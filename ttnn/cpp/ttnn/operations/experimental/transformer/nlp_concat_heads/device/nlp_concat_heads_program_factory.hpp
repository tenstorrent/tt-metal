// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "nlp_concat_heads_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::nlp_concat_heads::program {

struct NLPConcatHeadsSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    tt::tt_metal::CBHandle cb_src0 = 0;
    tt::tt_metal::CBHandle cb_out = 0;
    std::vector<CoreCoord> cores;
    bool in_sharded = false;
    bool out_sharded = false;
};

struct NLPConcatHeadsProgramFactory {
    using shared_variables_t = NLPConcatHeadsSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const NlpConcatHeadsParams& operation_attributes,
        const NlpConcatHeadsInputs& tensor_args,
        tensor_return_value_t& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const NlpConcatHeadsParams& operation_attributes,
        const NlpConcatHeadsInputs& tensor_args,
        tensor_return_value_t& output);
};

}  // namespace ttnn::operations::experimental::nlp_concat_heads::program
