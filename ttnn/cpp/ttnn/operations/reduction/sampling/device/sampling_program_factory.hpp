// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal2_artifacts.hpp"
#include "ttnn/operations/reduction/sampling/device/sampling_device_operation_types.hpp"

namespace ttnn::prim {

struct SamplingProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_spec(
        const SamplingParams& operation_attributes, const SamplingInputs& tensor_args, Tensor& output_tensor);
};

}  // namespace ttnn::prim
