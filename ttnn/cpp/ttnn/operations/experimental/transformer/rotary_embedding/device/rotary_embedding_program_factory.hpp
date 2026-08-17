// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/transformer/rotary_embedding/device/rotary_embedding_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

struct RotaryEmbeddingProgramFactory {
    // Contract (1): single ProgramDescriptor.  Sharded variants set CBDescriptor::buffer.  Cache-hit
    // re-application is owned by RotaryEmbeddingDeviceOperation::override_runtime_arguments, which
    // patches the addresses and decode scalars in place -- this is a cache-miss-only path.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const RotaryEmbeddingParams& operation_attributes,
        const RotaryEmbeddingInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
