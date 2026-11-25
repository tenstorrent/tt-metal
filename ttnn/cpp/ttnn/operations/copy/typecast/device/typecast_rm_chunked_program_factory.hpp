// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "typecast_device_op_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::prim {

struct TypecastRowMajorChunkedProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output);

    // Rewrites the row counts and start rows on a hit at another shape, and rebinds the tensors.
    static tt::tt_metal::ProgramRunArgs override_runtime_arguments(
        const TypecastParams& args,
        const TypecastInputs& tensor_args,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

// Whether a row-major program can serve another shape with the same row width and rank: with runtime_shapes, a
// row-major interleaved input on the whole grid, an interleaved output, no preallocated output and a non-empty input.
// Its kernels take the row counts at run time, so only the row width (the page size) and, below one row per core, the
// row count shape the program.
bool typecast_rm_chunked_is_shape_free(const TypecastParams& args, const TypecastInputs& tensor_args);

}  // namespace ttnn::prim
