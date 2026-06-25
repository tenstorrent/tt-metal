// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>
#include "ttnn/device_operation.hpp"
#include "rotary_embedding_llama_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct RotaryEmbeddingLlamaMultiCoreSharded {
    // Metal 2.0 (MetalV2FactoryConcept): single ProgramSpec for the fully-sharded decode
    // case. All five working tensors (input/cos/sin/trans_mat/output) back borrowed-memory
    // DFBs via DataflowBufferSpec::borrowed_from, so the framework patches dynamic addresses
    // on cache hit (matching the legacy UpdateDynamicCircularBufferAddress chain). The op is
    // compute-only, so every DFB is one-ended from the validator's view and is bound on the
    // single compute kernel as an INTRA self-loop (PRODUCER + CONSUMER).
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const RotaryEmbeddingLlamaParams& operation_attributes,
        const RotaryEmbeddingLlamaInputs& tensor_args,
        tt::tt_metal::Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
