// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gelu_bw_device_operation_types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::unary_backward::gelu_bw {

struct GeluBwProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const GeluBwParams& args, const GeluBwInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::operations::unary_backward::gelu_bw
