// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/transformer/rotary_embedding/device/rotary_embedding_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

struct RotaryEmbeddingProgramFactory {
    // Contract (1): single ProgramDescriptor.  Two variants: single-tile (Wt == 1) and multi-tile.
    // Sharded variants set CBDescriptor::buffer for the globally-allocated input/output CBs.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const RotaryEmbeddingParams& operation_attributes,
        const RotaryEmbeddingInputs& tensor_args,
        Tensor& tensor_return_value);

    // Patches the cached program in place on every cache hit: the token_idx-derived cos/sin offsets
    // (its value is excluded from the hash) plus every buffer address and globally-allocated CB
    // address, since this hook supersedes resolve_bindings.  Does NOT rebuild the descriptor.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const RotaryEmbeddingParams& operation_attributes,
        const RotaryEmbeddingInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::experimental::prim
