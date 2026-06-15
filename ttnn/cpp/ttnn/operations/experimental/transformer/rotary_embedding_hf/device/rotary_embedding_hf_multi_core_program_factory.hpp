// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/transformer/rotary_embedding_hf/device/rotary_embedding_hf_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

struct RotaryEmbeddingHfMultiCore {
    // Contract (1): single ProgramDescriptor.  Sharded variants set CBDescriptor::buffer
    // so the framework patches the dynamic address on cache hit.  Per-core reader/writer
    // runtime args are re-applied by the framework via apply_descriptor_runtime_args.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const RotaryEmbeddingHfParams& operation_attributes,
        const RotaryEmbeddingHfInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::experimental::prim
