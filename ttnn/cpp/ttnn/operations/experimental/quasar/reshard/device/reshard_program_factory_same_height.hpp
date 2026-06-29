// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/program_spec_artifacts.hpp"
#include "ttnn/operations/experimental/quasar/reshard/device/reshard_device_operation_types.hpp"

namespace ttnn::prim::qsr {

// WIDTH_SHARDED -> WIDTH_SHARDED reshard.
//
// Metal 2.0 host API factory: returns ProgramSpecArtifacts (ProgramSpec + ProgramRunArgs)
// rather than a ProgramDescriptor. Satisfies ProgramSpecFactoryConcept.
template <bool local_is_output>
struct ReshardSameHeightFactory {
    static ttnn::device_operation::ProgramSpecArtifacts create_program_spec(
        const ReshardParams& operation_attributes, const ReshardInputs& tensor_args, Tensor& output_tensor);
};

}  // namespace ttnn::prim::qsr
