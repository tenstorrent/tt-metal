// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "qkv_causal_conv1d_silu_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct QkvCausalConv1dSiluProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const QkvCausalConv1dSiluParams&, const QkvCausalConv1dSiluInputs&, std::vector<Tensor>&);
};

}  // namespace ttnn::experimental::prim
