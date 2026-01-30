// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_vit/device/nlp_create_qkv_heads_vit_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct NlpCreateQkvHeadsVitSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    uint32_t num_cores = 0;
    uint32_t num_cores_y = 0;
};

struct NlpCreateQkvHeadsVitProgramFactory {
    using shared_variables_t = NlpCreateQkvHeadsVitSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const NlpCreateQkvHeadsVitParams& operation_attributes,
        const NlpCreateQkvHeadsVitInputs& tensor_args,
        NlpCreateQkvHeadsVitResult& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const NlpCreateQkvHeadsVitParams& operation_attributes,
        const NlpCreateQkvHeadsVitInputs& tensor_args,
        NlpCreateQkvHeadsVitResult& output);
};

}  // namespace ttnn::experimental::prim
