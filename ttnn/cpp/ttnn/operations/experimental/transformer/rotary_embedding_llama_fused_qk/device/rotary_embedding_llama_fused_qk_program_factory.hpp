// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk/device/rotary_embedding_llama_fused_qk_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct RotaryEmbeddingLlamaFusedQKProgramFactory {
    // Metal 2.0 (MetalV2FactoryConcept): a single ProgramSpec + ProgramRunArgs.  All seven working
    // tensors (q/k inputs, cos/sin/trans_mat, q/k outputs) are sharded and bind through
    // borrowed-memory DFBs (DataflowBufferSpec::borrowed_from) so the framework patches dynamic
    // addresses on cache hit.  The single compute kernel takes one per-core runtime arg (is_q) to
    // select q vs k work, and its source file is selected at runtime by row_major_QK.
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const RotaryEmbeddingLlamaFusedQkParams& operation_attributes,
        const RotaryEmbeddingLlamaFusedQkInputs& tensor_args,
        RotaryEmbeddingLlamaFusedQkResult& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
