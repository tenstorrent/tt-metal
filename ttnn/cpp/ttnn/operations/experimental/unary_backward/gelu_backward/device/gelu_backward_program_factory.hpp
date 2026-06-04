// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gelu_backward_device_operation_types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

struct GeluBackwardProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const GeluBackwardParams& args, const GeluBackwardInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::experimental::prim
