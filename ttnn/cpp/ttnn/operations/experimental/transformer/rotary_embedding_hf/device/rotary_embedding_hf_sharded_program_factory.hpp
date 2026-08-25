// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/transformer/rotary_embedding_hf/device/rotary_embedding_hf_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct RotaryEmbeddingHfMultiCoreSharded {
    // Metal 2.0 ProgramSpecFactoryConcept. All four working DFBs (input/cos/sin/output)
    // are sharded — they are built on borrowed memory (DataflowBufferSpec::borrowed_from
    // naming the io TensorParameter), so the framework refreshes their backing addresses
    // from the tensor arguments on every cache hit.
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const RotaryEmbeddingHfParams& operation_attributes,
        const RotaryEmbeddingHfInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::experimental::prim
