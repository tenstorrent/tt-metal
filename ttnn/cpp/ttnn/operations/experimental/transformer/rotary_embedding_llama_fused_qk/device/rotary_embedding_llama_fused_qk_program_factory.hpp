// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk/device/rotary_embedding_llama_fused_qk_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::transformer::rotary_embedding_llama_fused_qk::program {

struct RotaryEmbeddingLlamaFusedQKSharedVariables {
    tt::tt_metal::CBHandle cb_q_input{};
    tt::tt_metal::CBHandle cb_k_input{};
    tt::tt_metal::CBHandle cb_cos{};
    tt::tt_metal::CBHandle cb_sin{};
    tt::tt_metal::CBHandle cb_trans_mat{};
    tt::tt_metal::CBHandle cb_q_output{};
    tt::tt_metal::CBHandle cb_k_output{};
};

struct RotaryEmbeddingLlamaFusedQKProgramFactory {
    using shared_variables_t = RotaryEmbeddingLlamaFusedQKSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const RotaryEmbeddingLlamaFusedQkParams& operation_attributes,
        const RotaryEmbeddingLlamaFusedQkInputs& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const RotaryEmbeddingLlamaFusedQkParams& operation_attributes,
        const RotaryEmbeddingLlamaFusedQkInputs& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::transformer::rotary_embedding_llama_fused_qk::program
