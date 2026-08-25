// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding/device/rotary_embedding_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::experimental::prim {

struct RotaryEmbeddingProgramFactory {
    // Metal 2.0 spec factory: builds a ProgramSpec + ProgramRunArgs declaratively (cache-miss-only
    // path; the framework owns program construction and caching). Two internal variants selected by
    // shape: single-tile (Wt == 1) and multi-tile. Sharded variants back the input/output DFBs with
    // borrowed tensor memory.
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const RotaryEmbeddingParams& operation_attributes,
        const RotaryEmbeddingInputs& tensor_args,
        Tensor& tensor_return_value);

    // CustomProgramSpecFactoryConcept cache-hit hook: returns the per-dispatch ProgramRunArgs the
    // framework applies via UpdateProgramRunArgs. Decode mode (token_idx set) derives
    // cos_sin_start_id / cos_sin_offset from token_idx, while token_idx is deliberately excluded from
    // compute_program_hash so successive decode positions cache-hit the same program. Those two
    // scalars must therefore be re-applied on every cache hit -- otherwise the cached program keeps
    // the first token's offsets and every later token reads the wrong cos/sin rows. On this concept
    // the framework refreshes nothing on its own, so the hook also re-binds every io tensor (their
    // addresses -- runtime args and borrowed-DFB backing memory alike -- refresh through the typed
    // tensor channel).
    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const RotaryEmbeddingParams& operation_attributes,
        const RotaryEmbeddingInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::experimental::prim
