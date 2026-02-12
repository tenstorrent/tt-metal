// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "nlp_create_qkv_heads_falcon7b_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct NlpCreateQkvHeadsFalcon7BSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    uint32_t num_cores = 0;
    uint32_t num_cores_y = 0;
};

struct NlpCreateQkvHeadsFalcon7BProgramFactory {
    using shared_variables_t = NlpCreateQkvHeadsFalcon7BSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const NlpCreateQkvHeadsFalcon7bParams& operation_attributes,
        const Tensor& tensor_args,
        NlpCreateQkvHeadsFalcon7bResult& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const NlpCreateQkvHeadsFalcon7bParams& operation_attributes,
        const Tensor& tensor_args,
        NlpCreateQkvHeadsFalcon7bResult& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
