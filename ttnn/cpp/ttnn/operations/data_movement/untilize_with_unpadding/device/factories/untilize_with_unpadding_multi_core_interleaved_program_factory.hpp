// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/data_movement/untilize_with_unpadding/device/untilize_with_unpadding_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::prim {

struct UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output);

    // Rewrites every runtime arg, which follows the shapes, and rebinds the tensors. A shape-free program
    // (untilize_with_unpadding_interleaved_is_shape_free) serves other shapes with the same last dim this way.
    static tt::tt_metal::ProgramRunArgs override_runtime_arguments(
        const UntilizeWithUnpaddingParams& operation_attributes,
        const Tensor& input,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

// Whether the multi-core interleaved program serves every shape with the same last dim: runtime_shapes set,
// interleaved input and output, at most 4D.
bool untilize_with_unpadding_interleaved_is_shape_free(
    const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input);

}  // namespace ttnn::prim
