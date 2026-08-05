// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/device_operation.hpp"
#include "rotary_embedding_llama_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct RotaryEmbeddingLlamaMultiCoreSharded {
    // Contract (1): single ProgramDescriptor for the fully-sharded decode case.
    // All five working CBs (input/cos/sin/trans_mat/output) bind through
    // CBDescriptor::buffer so the framework patches dynamic addresses on cache hit,
    // matching the legacy UpdateDynamicCircularBufferAddress chain.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const RotaryEmbeddingLlamaParams& operation_attributes,
        const RotaryEmbeddingLlamaInputs& tensor_args,
        ttnn::Tensor& output);
};

}  // namespace ttnn::experimental::prim
