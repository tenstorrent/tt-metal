// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>
#include "ttnn/device_operation.hpp"
#include "rotary_embedding_llama_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct RotaryEmbeddingLlamaMultiCore {
    // Metal 2.0 (MetalV2FactoryConcept): ProgramSpec for the interleaved (non-sharded) prefill
    // case. reader -> compute -> writer pipeline; the five tensors are Case 1 (TensorAccessor
    // bindings on the reader/writer). Per-core reader/writer/compute runtime args
    // {batch_start,batch_end,seq_t_start,seq_t_end} are named RTAs; idle cores get zero-filled
    // args so they don't wait on cos/sin data that never arrives.
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const RotaryEmbeddingLlamaParams& operation_attributes,
        const RotaryEmbeddingLlamaInputs& tensor_args,
        tt::tt_metal::Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
