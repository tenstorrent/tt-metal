// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kda_causal_conv_device_operation_types.hpp"

namespace ttnn::prim {

struct KdaCausalConvProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const KdaCausalConvParams&, const KdaCausalConvInputs&, std::vector<Tensor>&);
};

}  // namespace ttnn::prim
