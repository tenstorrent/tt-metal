// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "rotary_embedding_llama_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct RotaryEmbeddingLlamaMultiCoreSharded {
    // Metal 2.0 factory (MetalV2FactoryConcept) for the fully-sharded decode case. All five io
    // buffers (input/cos/sin/trans_mat/output) bind through borrowed-memory DataflowBuffers
    // (DataflowBufferSpec::borrowed_from) so their L1 addresses resolve from the tensor args on each
    // cache hit, matching the legacy dynamic-address update chain.
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const RotaryEmbeddingLlamaParams& operation_attributes,
        const RotaryEmbeddingLlamaInputs& tensor_args,
        ttnn::Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
