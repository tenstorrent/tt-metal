// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk/device/rotary_embedding_llama_fused_qk_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct RotaryEmbeddingLlamaFusedQKProgramFactory {
    // Metal 2.0 (ProgramSpecFactoryConcept): single ProgramSpec for the compute-only op.
    // All seven working CBs (q/k inputs, cos/sin/trans_mat, q/k outputs) are sharded and bind
    // through DataflowBufferSpec::borrowed_from so the framework patches dynamic addresses on
    // cache hit. The single compute kernel takes one per-core runtime arg to select q vs k work.
    static ttnn::device_operation::ProgramArtifacts create_program_spec(
        const RotaryEmbeddingLlamaFusedQkParams& operation_attributes,
        const RotaryEmbeddingLlamaFusedQkInputs& tensor_args,
        RotaryEmbeddingLlamaFusedQkResult& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
