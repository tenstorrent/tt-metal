// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sigmoid_gated_rms_norm_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct SigmoidGatedRmsNormProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SigmoidGatedRmsNormParams&, const SigmoidGatedRmsNormInputs&, std::vector<Tensor>&);
};

}  // namespace ttnn::experimental::prim
