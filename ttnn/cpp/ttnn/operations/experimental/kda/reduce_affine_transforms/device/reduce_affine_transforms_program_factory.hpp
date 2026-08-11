// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "reduce_affine_transforms_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct ReduceAffineTransformsProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ReduceAffineTransformsParams&, const ReduceAffineTransformsInputs&, std::vector<Tensor>&);
};

}  // namespace ttnn::experimental::prim
