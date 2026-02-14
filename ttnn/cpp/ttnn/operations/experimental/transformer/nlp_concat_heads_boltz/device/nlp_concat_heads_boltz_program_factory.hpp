// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "nlp_concat_heads_boltz_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct NLPConcatHeadsBoltzSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    std::vector<CoreCoord> cores;
    tt::tt_metal::CBHandle cb_src0{};
    tt::tt_metal::CBHandle cb_out{};
};

struct NLPConcatHeadsBoltzProgramFactory {
    using shared_variables_t = NLPConcatHeadsBoltzSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const NLPConcatHeadsBoltzParams& operation_attributes,
        const NLPConcatHeadsBoltzInputs& tensor_args,
        Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const NLPConcatHeadsBoltzParams& operation_attributes,
        const NLPConcatHeadsBoltzInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::experimental::prim
