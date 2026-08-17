// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "rotary_embedding_llama_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct RotaryEmbeddingLlamaMultiCorePrefillSharded {
    // Metal 2.0 factory (MetalV2FactoryConcept) for prefill with sharded cos/sin/trans_mat.
    // Globally-allocated (L1-resident) cos/sin/trans_mat bind through borrowed-memory DataflowBuffers
    // (DataflowBufferSpec::borrowed_from) only when the shard grid covers all work-unit cores; otherwise
    // they are read via TensorAccessor. The work unit is placed on all device cores.
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const RotaryEmbeddingLlamaParams& operation_attributes,
        const RotaryEmbeddingLlamaInputs& tensor_args,
        ttnn::Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
