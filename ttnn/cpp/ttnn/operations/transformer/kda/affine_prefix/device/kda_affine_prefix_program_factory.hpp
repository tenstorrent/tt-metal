// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kda_affine_prefix_device_operation_types.hpp"

namespace ttnn::prim {

struct KdaAffinePrefixProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const KdaAffinePrefixParams&, const KdaAffinePrefixInputs&, std::vector<Tensor>&);
};

}  // namespace ttnn::prim
