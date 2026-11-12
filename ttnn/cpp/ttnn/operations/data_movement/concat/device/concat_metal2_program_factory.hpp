// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// [Quasar bring-up: concat interleaved metal2]

#pragma once

#include <optional>

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

#include "concat_device_operation_types.hpp"

namespace ttnn::prim {

// Metal-2.0 twin of ConcatProgramFactory (the generic interleaved-output concat).
//
// Quasar cannot build the legacy DataMovementKernel that ConcatProgramFactory's ProgramDescriptor
// emits (kernel.hpp:449-451, "DataMovementKernel is not supported on Quasar"), so on Quasar
// ConcatDeviceOperation::select_program_factory routes here instead. WH/BH keep the legacy
// descriptor path byte-for-byte. Same dual-factory + arch-gated-selector shape that
// reshape_view already uses.
struct ConcatMetalV2ProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ConcatParams& operation_attributes, const ConcatInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
