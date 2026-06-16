// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/metal2_artifacts.hpp"
#include "tilize_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {
// Metal 2.0 program factory: returns a ProgramSpec + ProgramRunArgs artifact
// (ProgramSpecFactoryConcept). The other tilize factories remain on the legacy
// ProgramDescriptor concept; the framework adapter dispatches per-factory.
struct TilizeSingleCoreProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_spec(
        const TilizeParams& operation_attributes, const TilizeInputs& tensor_args, Tensor& tensor_return_value);
};
}  // namespace ttnn::prim
