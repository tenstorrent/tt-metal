// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tanh_bw_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::unary_backward::tanh_bw {

struct TanhBwProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const TanhBwParams& args, const TanhBwInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::operations::unary_backward::tanh_bw
