// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/experimental/quasar/reshape_view/device/reshape_device_operation_types.hpp"

namespace ttnn::prim::qsr {

// Legacy (Gen1 / WH,BH) row-major reshape: emits a ProgramDescriptor with DataMovementKernels.
struct ReshapeViewRMProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ReshapeViewParams& operation_attributes,
        const ReshapeViewInputs& tensor_args,
        Tensor& tensor_return_value);
};

// Metal-2.0 (Quasar) row-major reshape: Quasar cannot build a legacy DataMovementKernel
// (kernel.hpp:382), so on Quasar we emit a ProgramSpec + QuasarDataMovementKernel instead.
struct ReshapeViewRMMetalV2ProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ReshapeViewParams& operation_attributes,
        const ReshapeViewInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim::qsr
