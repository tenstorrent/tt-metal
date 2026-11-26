// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct TilizeWithValPaddingMultiCoreDefaultFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const TilizeWithValPaddingParams& operation_attributes,
        const Tensor& input_tensor,
        Tensor& tensor_return_value);

    // Rebinds the tensors, and for a shape-free program (tilize_with_val_padding_interleaved_is_shape_free)
    // rewrites every runtime arg, which then follows the shapes.
    static tt::tt_metal::ProgramRunArgs override_runtime_arguments(
        const TilizeWithValPaddingParams& operation_attributes,
        const Tensor& input_tensor,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

// Whether the multi-core interleaved program serves every shape with the same last dims: runtime_shapes set,
// interleaved input and output, at most 4D.
bool tilize_with_val_padding_interleaved_is_shape_free(
    const TilizeWithValPaddingParams& operation_attributes, const Tensor& input_tensor);

}  // namespace ttnn::prim
