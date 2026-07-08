// SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/transformer/rotary_embedding_fused_qk/device/rotary_embedding_fused_qk_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct RotaryEmbeddingFusedQKSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    std::vector<CoreCoord> cores;
    uint32_t Wt = 0;
    uint32_t num_q_rows = 0;
    uint32_t num_k_rows = 0;
};

struct RotaryEmbeddingFusedQKProgramFactory {
    using shared_variables_t = RotaryEmbeddingFusedQKSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor>;

    static cached_program_t create(
        const RotaryEmbeddingFusedQKParams& operation_attributes,
        const RotaryEmbeddingFusedQKInputs& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const RotaryEmbeddingFusedQKParams& operation_attributes,
        const RotaryEmbeddingFusedQKInputs& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
