// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/operations/experimental/transformer/rotary_embedding/device/rotary_embedding_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

struct RotaryEmbeddingProgramFactory {
    // Contract (1): single ProgramDescriptor.  Sharded variants set CBDescriptor::buffer.  Cache-hit
    // re-application is owned by override_runtime_arguments below, which patches the addresses and
    // decode scalars in place -- this is a cache-miss-only path.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const RotaryEmbeddingParams& operation_attributes,
        const RotaryEmbeddingInputs& tensor_args,
        Tensor& tensor_return_value);

    // Decode mode (token_idx set) derives cos_sin_start_id / cos_sin_offset from token_idx and bakes
    // them into static reader/writer runtime args, while token_idx is deliberately excluded from
    // compute_program_hash so successive decode positions cache-hit the same program.  Those two
    // scalars must therefore be re-applied on every cache hit -- otherwise the cached program keeps
    // the first token's offsets and every later token reads the wrong cos/sin rows.  Declaring this
    // hook also supersedes resolve_bindings, so it owns buffer-address re-application (runtime args and
    // sharded CBs) too.  It patches those slots in place; it never rebuilds the descriptor.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const RotaryEmbeddingParams& operation_attributes,
        const RotaryEmbeddingInputs& tensor_args,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::experimental::prim
