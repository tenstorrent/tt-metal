// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/experimental/quasar/reshape_view/device/reshape_device_operation_types.hpp"

namespace ttnn::prim::qsr {

// ROW_MAJOR reshape: the flat byte stream is re-paged by one multi-hart DM kernel per node (no
// DataflowBuffer). Input and output are the rank-2 [rows, last_dim] views the host entry builds.
struct ReshapeViewRowMajorProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ReshapeViewParams& operation_attributes,
        const ReshapeViewInputs& tensor_args,
        Tensor& tensor_return_value);
};

// TILE reshape: one multi-hart DM kernel per node derives the output-tile <- input-tile element mapping
// itself and gathers every output tile with NOC reads (no DataflowBuffer, no mapping tensor). Input and
// output are the rank-3 [B, H, W] views the host entry builds.
struct ReshapeViewTiledProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ReshapeViewParams& operation_attributes,
        const ReshapeViewInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim::qsr
