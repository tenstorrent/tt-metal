// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/experimental/quasar/reshard/device/reshard_device_operation_types.hpp"

namespace ttnn::prim::qsr {

// General reshard (arbitrary sharding layouts).
//
// Metal 2.0 host API factory: returns ProgramArtifacts (ProgramSpec + ProgramRunArgs)
// rather than a ProgramDescriptor. Satisfies MetalV2FactoryConcept.
struct ReshardGenericFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ReshardParams& operation_attributes, const ReshardInputs& tensor_args, Tensor& output_tensor);
};

}  // namespace ttnn::prim::qsr
