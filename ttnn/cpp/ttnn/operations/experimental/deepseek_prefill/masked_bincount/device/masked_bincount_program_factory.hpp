// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "masked_bincount_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

struct MaskedBincountProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const MaskedBincountParams& operation_attributes,
        const MaskedBincountInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
