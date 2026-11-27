// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/data_movement/reshape_view/device/reshape_device_operation_types.hpp"

namespace ttnn::prim {

struct ReshapeViewTiledProgramFactory {
    // create_program_artifacts() materializes the host-computed input-to-output
    // page-mapping tensor onto the device and carries the owning MeshTensor in the
    // artifact's op_owned_tensors, so its backing buffer outlives the cached program.
    // The mapping is fully determined by the hashed input/output shapes.
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ReshapeViewParams& operation_attributes,
        const ReshapeViewInputs& tensor_args,
        Tensor& tensor_return_value);
};

// A tiled reshape that keeps the last dim: [A, R_in, C] to [B, R_out, C] for rank-3 tensors with standard
// 32x32 tiles, each output row being one input row in order. The kernels compute which input rows each output
// tile takes from the shapes, passed as runtime args, so it needs no host-built page map and no DRAM for one,
// and its binaries do not depend on the shapes. reshape.cpp's native_takes_row_factory keeps routing away from
// codegen exactly the calls this accepts; change both together.
bool reshape_view_tiled_keeps_last_dim(const ReshapeViewParams& operation_attributes, const Tensor& input);

struct ReshapeViewTiledRowsProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ReshapeViewParams& operation_attributes,
        const ReshapeViewInputs& tensor_args,
        Tensor& tensor_return_value);

    // Rebinds the tensors, and with reshape_view_tiled_rows_is_shape_free recomputes every core's args.
    static tt::tt_metal::ProgramRunArgs override_runtime_arguments(
        const ReshapeViewParams& operation_attributes,
        const ReshapeViewInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

// Whether a row-factory program serves every shape with the same last dim: with runtime_shapes, interleaved input and
// output and no sub core grid. Only the core set then still follows the shape, through
// reshape_view_tiled_rows_cores_key.
bool reshape_view_tiled_rows_is_shape_free(const ReshapeViewParams& operation_attributes, const Tensor& input);

// 0 when every core of the grid gets output tiles, otherwise the output tile count, which sets the core set.
uint32_t reshape_view_tiled_rows_cores_key(const ReshapeViewParams& operation_attributes, const Tensor& input);

}  // namespace ttnn::prim
