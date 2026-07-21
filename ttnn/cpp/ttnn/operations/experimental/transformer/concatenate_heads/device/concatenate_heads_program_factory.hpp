// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "concatenate_heads_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

struct ConcatenateHeadsProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ConcatenateHeadsParams& operation_attributes, const ConcatenateHeadsInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::experimental::prim
