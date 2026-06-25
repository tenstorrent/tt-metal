// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "transpose_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/program_spec_artifacts.hpp"

namespace ttnn::prim::qsr {

struct TransposeHCTiledInterleavedProgramFactory {
    static ttnn::device_operation::ProgramSpecArtifacts create_program_spec(
        const TransposeParams& operation_attributes, const TransposeInputs& tensor_args, Tensor& output_tensor);
};

}  // namespace ttnn::prim::qsr
