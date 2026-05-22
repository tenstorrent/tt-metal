// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "gelu_bw_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::unary_backward::gelu_bw {

struct GeluBwProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const GeluBwParams& args, const GeluBwInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::operations::unary_backward::gelu_bw
