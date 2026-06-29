// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/program_spec_artifacts.hpp"
#include "tilize_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim::qsr {

struct TilizeMultiCoreWidthShardedProgramFactory {
    static ttnn::device_operation::ProgramSpecArtifacts create_program_spec(
        const TilizeParams& operation_attributes, const TilizeInputs& tensor_args, Tensor& tensor_return_value);
};
}  // namespace ttnn::prim::qsr
