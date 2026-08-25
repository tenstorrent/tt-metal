// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/transformer/rotary_embedding_hf/device/rotary_embedding_hf_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct RotaryEmbeddingHfMultiCore {
    // Metal 2.0 ProgramSpecFactoryConcept. Prefill path; internally selects a single-tile
    // (head_dim == TILE_WIDTH) or multi-tile shape. When the input/output is sharded, the
    // corresponding DFB is built on borrowed memory (DataflowBufferSpec::borrowed_from), so
    // the framework refreshes its backing address from the tensor arguments on cache hits.
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const RotaryEmbeddingHfParams& operation_attributes,
        const RotaryEmbeddingHfInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::experimental::prim
