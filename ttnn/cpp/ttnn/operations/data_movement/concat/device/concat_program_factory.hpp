// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "concat_device_operation_types.hpp"

#include <optional>

#include <tt-metalium/workload/run_args/program_run_args.hpp>

#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::prim {

// Whether a ConcatProgramFactory program can serve another row count: a TILE concat of
// interleaved tensors along the rows (dim -2) on the full grid, with at least one output tile per
// core. The core set is then the whole grid at every row count, and everything that follows the
// rows is a runtime argument, so the program hash leaves the rows out.
bool concat_program_is_row_free(const ConcatParams& operation_attributes, const ConcatInputs& tensor_args);

struct ConcatProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ConcatParams& operation_attributes, const ConcatInputs& tensor_args, Tensor& tensor_return_value);

    // Rewrites the per-core page counts and start pages, and rebinds the tensors.
    static tt::tt_metal::ProgramRunArgs override_runtime_arguments(
        const ConcatParams& operation_attributes,
        const ConcatInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::prim
