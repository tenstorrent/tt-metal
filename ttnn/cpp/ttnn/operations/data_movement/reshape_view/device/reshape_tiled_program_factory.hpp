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
};

}  // namespace ttnn::prim
