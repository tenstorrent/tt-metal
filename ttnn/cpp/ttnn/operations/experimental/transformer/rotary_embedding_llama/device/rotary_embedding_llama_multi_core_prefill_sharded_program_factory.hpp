// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/device_operation.hpp"
#include "rotary_embedding_llama_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct RotaryEmbeddingLlamaMultiCorePrefillSharded {
    // Contract (1): single ProgramDescriptor.  Globally-allocated cos/sin/trans_mat
    // CBs (when sharded and grid-covering) bind through CBDescriptor::buffer so the
    // framework patches dynamic addresses on cache hit; partial-grid shard cases
    // get a second non-buffered CB on the remaining cores (CBs are merged at
    // runtime since they share buffer indices).
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const RotaryEmbeddingLlamaParams& operation_attributes,
        const RotaryEmbeddingLlamaInputs& tensor_args,
        ttnn::Tensor& output);
};

}  // namespace ttnn::experimental::prim
