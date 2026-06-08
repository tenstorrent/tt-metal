// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/device_operation.hpp"
#include "rotary_embedding_llama_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct RotaryEmbeddingLlamaMultiCore {
    // Contract (1): single ProgramDescriptor for the interleaved (non-sharded) case.
    // Per-core reader/writer/compute runtime args are re-applied by the framework via
    // apply_descriptor_runtime_args; idle cores get zero-filled args so they don't wait
    // on cos/sin data that never arrives.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const RotaryEmbeddingLlamaParams& operation_attributes,
        const RotaryEmbeddingLlamaInputs& tensor_args,
        tt::tt_metal::Tensor& output);
};

}  // namespace ttnn::experimental::prim
