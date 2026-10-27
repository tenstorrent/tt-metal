// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/experimental/quasar/reshape_view/device/reshape_device_operation_types.hpp"

namespace ttnn::prim::qsr {

// Metal-2.0 (Quasar) tiled reshape: Quasar cannot build a legacy DataMovementKernel
// (kernel.hpp:382), so on Quasar we emit a ProgramSpec + QuasarDataMovementKernel instead
// of the legacy ProgramDescriptor / WorkloadDescriptor path. Reproduces the same host-computed
// input->output tile page-mapping data movement as ReshapeViewTiledProgramFactory.
struct ReshapeViewTiledMetalV2ProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ReshapeViewParams& operation_attributes,
        const ReshapeViewInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim::qsr
