// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/transformer/rotary_embedding/device/rotary_embedding_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

struct RotaryEmbeddingProgramFactory {
    // Contract (1): single ProgramDescriptor.  Sharded variants set CBDescriptor::buffer
    // so the framework patches dynamic CB addresses on cache hit; per-core runtime args
    // and CB total_sizes (which depend on input shape) are re-applied via
    // apply_descriptor_runtime_args.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const RotaryEmbeddingParams& operation_attributes,
        const RotaryEmbeddingInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
