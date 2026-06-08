// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/transformer/rotary_embedding_hf/device/rotary_embedding_hf_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

struct RotaryEmbeddingHfMultiCoreSharded {
    // Contract (1): single ProgramDescriptor.  All four working CBs (input/cos/sin/output)
    // are sharded — they use CBDescriptor::buffer so the framework patches dynamic
    // addresses on cache hit, mirroring the legacy UpdateDynamicCircularBufferAddress chain.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const RotaryEmbeddingHfParams& operation_attributes,
        const RotaryEmbeddingHfInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::experimental::prim
