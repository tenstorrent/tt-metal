// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kda_gated_rms_device_operation_types.hpp"

namespace ttnn::prim {

struct KdaGatedRmsProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const KdaGatedRmsParams&, const KdaGatedRmsInputs&, std::vector<Tensor>&);
};

}  // namespace ttnn::prim
